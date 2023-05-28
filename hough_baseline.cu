//////////////////////////////////////////////////////////////////////////////////////////////////////////
// ECE 759 Project : Implementation of Hough Transform on Nvidia GPU
//
// The following open source codebases were used to help the development of the project:
//
// https://rosettacode.org/w/index.php?title=Canny_edge_detector&oldid=329226 (last visited December 10, 2022) under CC BY-SA 4.0 License
// https://github.com/nothings/stb.git (last visited December 10, 2022) under MIT License 
// https://github.com/eToTheEcs/hough-transform (last visited December 14, 2022)  -> Code used as a reference model for initial debugging of the algorithm.



#include <iostream>
#include <cmath>
#include <vector>
#include <list>
#include <utility>
#include <queue>
#include <cstring>
#include <cuda.h>
#include <time.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "canny.h"



using namespace std;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

typedef unsigned char uchar;

#define PI 3.141592653589793238


#define BIN_WIDTH 1                
#define NUM_BINS 180 / BIN_WIDTH   

//Parameters for Canny Edge detection//

#define KERNEL_SIZE 3

#define RATIO 3



void color_index(uchar* input, int i, int j, int w) {


	input[3*(i*w +j)] = 255;
	input[3*(i*w +j)+1] =0;
	input[3*(i*w +j)+2] = 0;

}	

void draw_line(uchar* input, int w, int h, pair<int, int> p1, pair<int, int> p2) {

int x1, x2, y1, y2;
int thickness = 2;
float gran_x=0.01;
float minx, maxx;
float slope, c;

	x1= p1.first;
	y1= p1.second;
	x2= p2.first;
	y2= p2.second;
	
	slope =   (float)(y2-y1)/(x2-x1);
	c= y1 - slope*x1;




	
	if(x1!=x2) {


		if(x1<x2) {
			minx=(float)x1;
			maxx=(float)x2;
		} else {
			minx=(float)x2;
			maxx=(float)x1;
		}

		float iter;
		int i; //row number 
		int j; //column number

		if(minx<0) {
		
			minx= 0;
		}

		if(maxx>w) {
			maxx=w;
		}
			
		iter= minx;
	

        	
		while(iter < maxx) {

			i= round(c + slope*iter);
			j= round(iter);

			if((i>0)&&(i<h)) {
				color_index(input, i, j, w);
			}

			iter= iter+ gran_x;
			
		}
	} else {
		int maxy, miny;

		if(y2>y1){
			miny=y1;
			maxy=y2;
		} else {
			miny=y2;
			maxy=y1;
		}
	
		if(miny<0) {
			miny= 0;
		}
		if(maxy>h) {
			maxy=h;
		}

		for(int i=miny; i<maxy; i++) {
			color_index(input, i, x1, w);
		}


	}
}




void canny_detector(const uchar* input, const int w, const int h, const  int threshold, uchar* output) {

	bitmap_info_header_t bmp_ih;
	pixel_t* input_int;
	pixel_t* output_int;

	
	input_int= new pixel_t [w*h];
	output_int = new pixel_t [w*h];

	for(int i=0; i< w; i++) {
		for(int j=0; j<h; j++) {
			input_int[j*w + i] = input[j*w + i];
		}
	}
	
	bmp_ih.width = w;
        bmp_ih.height = h;
	bmp_ih.bmp_bytesz = w*h;

	output_int = canny_edge_detection(input_int, &bmp_ih, threshold,  threshold*RATIO, 1); 

	for(int i=0; i< w; i++) {
		for(int j=0; j<h; j++) {
			output[j*w + i] = (uchar)output_int[j*w + i];
		}
	}

	
}




void polarToCartesian(int rho, int theta, pair<int,int> &p1, pair<int,int> &p2);

__global__ void voting_kernel(uchar * edge_array, int* hough_space,  int width, int height, int diag) {

	//Pixel indices
	int i, j;
	
	int rho;
	double theta_rad;
	

	int index = blockDim.x*blockIdx.x + threadIdx.x;

	i = index/width;
	j = index%width;

	if(index< width* height){
		if(*(edge_array + index) == 255) {
		       for(int theta = 0; theta <= 180; theta += BIN_WIDTH) {

				theta_rad = ((double)((double)(theta -90) /180)) * PI;
                       		rho = round(j * cos((double)(theta_rad)) + i * sin((double)(theta_rad))) + diag;

				//Atomically vote in the hough space
                       		atomicAdd((hough_space + rho*(NUM_BINS+1) + theta), 1);
			}
                }
			
	}
}
	

__global__ void equation_forming_kernel (int* hough_space, int input_length, int* rho_vals, int* theta_vals, int* output_length, int threshold, int diag) {

	
	extern __shared__ int s [];

	//Shared Memory Arrays
        int* arr_size = &s[0];

        int* rho_arr =  &s[1];   
        int* theta_arr =  &s[1+ blockDim.x];



        int rho, theta;
	int smem_array_size;
	
	

	int store_index;	

	int index = blockDim.x*blockIdx.x + threadIdx.x;
	
	if(threadIdx.x ==0) {
		//Shared Memory variables are always uninitialized, initialize it here:
 		*arr_size=0;
		
	}

	__syncthreads();

	if(index< input_length) {
		if(*(hough_space + index) > threshold) {
			rho =   index/(NUM_BINS+1);
			theta = index%(NUM_BINS+1);
			
			rho = rho - diag;
			theta = theta -90;
			
			store_index = atomicAdd(arr_size, 1);
				
			rho_arr[store_index] = rho ;
			theta_arr[store_index] = theta;
		 }
	}
			 
			
	__syncthreads();
	
	smem_array_size = *arr_size;
	
	if(threadIdx.x ==0) {
		*arr_size = atomicAdd(output_length, smem_array_size);
	}
        __syncthreads();
     
	if(threadIdx.x < smem_array_size){
		rho_vals[(*arr_size)+ threadIdx.x] = rho_arr[threadIdx.x];
		theta_vals[(*arr_size)+ threadIdx.x] = theta_arr[threadIdx.x];
	}
} 

	      	

	

	

void reference_serial_voting( const uchar* edges, int w, int h, int * hough_space)  {

 
    int rho;
    int theta;
    int diag;

    diag = hypot(h, w);
    double theta_rad;
 
    

    

    

    for(int i = 0; i < h; ++i) {
        for(int j = 0; j < w; ++j) {
	   
            if(edges[i*w+j] == 255) {  

		//In the Image:
                // We are taking theta from -90 to +90. Since negative indices are not allowed theta is offsetted to go from 0 to 180
		// The value of rho can go from (-max(WIDTH, HEIGHT), sqrt(WIDTH^2 + HEIGHT^2)) -> Taking (-sqrt(WIDTH^2, HEIGHT^2), sqrt(WIDTH^2, HEIGHT^2)) to be on safer side. -> Since negative indices are not alllowed rho here can go from 0 to 2*(sqrt(WIDTH^2 + HEIGHT^2)
                for(theta = 0; theta <= NUM_BINS; theta += BIN_WIDTH) {
		    theta_rad = ((double)(theta-90)/180) * PI;
                    rho = round(j * cos(theta_rad)  + i * sin(theta_rad)) + diag;
                    (*(hough_space + rho*(NUM_BINS +1) + theta))++;
                }
            }
        }
    }




}

void reference_equation_forming( int* hough_space ,   list<pair<int,int>> &equations,   int hough_size, int line_threshold) {

    pair<int,int> temp_pair;
    int rho, theta; 		
    int diag = hough_size / (2*(NUM_BINS+1));

   
    for(int i = 0; i < hough_size; ++i) {
       

            if((*(hough_space + i)) > line_threshold) {

		rho= i/(NUM_BINS+1);
 		theta = i%(NUM_BINS+1);
		
		//In the final equations theta -> (-90 to 90); rho -> (-maxdistance, maxdistance)
                rho = rho - diag;
                theta = theta - 90;
		temp_pair = pair<int, int>( rho, theta);
                equations.insert(equations.end(), temp_pair);
                
            }
     }
    
}

void initialize_to_zero ( int** arr, unsigned int size) {
	for( int i=0; i< size; i++) {
		*(*arr + i) =0;
	}
}


int main(int argc, char** argv) {

   
    int theta;      
    int canny_threshold;
    int rho;       
    int diag;
    int rho_max;

    

    cudaEvent_t start_cuda;
    cudaEvent_t stop_cuda;
    float ms;

    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);


    if(argc < 4) {
        printf("USAGE: hough [fileName] [treshold] [canny_threshold]\n");
        return EXIT_FAILURE;
    }

    int line_threshold = atoi(argv[2]);
    canny_threshold = atoi(argv[3]);

    
    ////////////////////////////////////////////////////////////
    ///////////////    Loading Input Image     /////////////////
    ////////////////////////////////////////////////////////////

    int h, w, inp_ch, desired_ch;

    uchar* image_src;
   
    desired_ch =1;

    
    image_src = stbi_load(argv[1], &w, &h, &inp_ch, desired_ch);


    cout<<"Image Loaded:: name= "<<argv[1]<<" w= "<<w<<" h= "<<h<<" inp_ch= "<< inp_ch<<endl; 

    
    (void)stbi_write_jpg("source.jpg",  w, h, 1, image_src, 100);

    //Calculating parameters from the dimensions of the image
    diag = hypot(h, w);
    rho_max = 2*diag;

    /////////////////////////////////////////////////////////////
    //////////////   Canny Edge Detection Stage  ////////////////
    /////////////////////////////////////////////////////////////

    uchar* edges;
    edges = new uchar [w*h];
    
    canny_detector(image_src, w, h, canny_threshold, edges);
        

    (void)stbi_write_jpg("edges.jpg",  w, h, 1, edges, 100);





    ///////////////////////////////////////////////////////////
    //////////// Reference Hough Space Computation/////////////
    ///////////////////////////////////////////////////////////

    int* hough_space_ref;   

    list<pair<int, int>> equations_ref;
    hough_space_ref = new int [ (NUM_BINS+1)* rho_max ]; 
   
    //Important:: Initialize accumulator matrix to 0
    initialize_to_zero(&hough_space_ref, (NUM_BINS+1)* rho_max); 


    start = high_resolution_clock::now();
    //Reference Serial implementation 
    reference_serial_voting( edges, w, h,  hough_space_ref);
    reference_equation_forming( hough_space_ref ,  equations_ref,  rho_max * (NUM_BINS+1), line_threshold);
    end = high_resolution_clock::now();
    

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end-start);
 
    cout<<fixed;
    cout<<"Time taken by serial implementation  "<<duration_sec.count()<<endl;
    
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////
    /////////////////////////////////////////////////////


    /////////////////////////////////////////////////////
    ////////     Code For GPU Execution     /////////////
    /////////////////////////////////////////////////////

    uchar* edges_host;
    uchar* edges_device;
    int* hough_space_device;
    int* rho_arr_device;
    int* theta_arr_device;
    int* rho_arr_host;
    int* theta_arr_host;
    int* eq_size_device;
    int* eq_size_host;
    int* hough_space_host;
    
    cudaError_t error_code;

    int  number_of_threads = 1024;
    int number_of_blocks_1 = (h * w + number_of_threads -1) /number_of_threads;
    int number_of_blocks_2 = (rho_max*(NUM_BINS+1) + number_of_threads-1) / number_of_threads;


    
    //Allocating GPU side memories
    cudaMalloc((void**)&edges_device, sizeof(uchar)*w*h);
    cudaMalloc((void**)&hough_space_device, sizeof(int)* rho_max * (NUM_BINS+1));
    cudaMalloc((void**)&theta_arr_device, sizeof(int)* rho_max * (NUM_BINS+1));
    cudaMalloc((void**)&rho_arr_device, sizeof(int) *rho_max *(NUM_BINS+1));
    cudaMalloc((void**)&eq_size_device, sizeof(int));


    //Allocating GPU side memories
    edges_host= new uchar [h*w];
    hough_space_host = new int [rho_max* (NUM_BINS+1)];
    rho_arr_host = new int [rho_max* (NUM_BINS+1)];
    theta_arr_host = new int [rho_max* (NUM_BINS+1)];
    eq_size_host = new int [rho_max *(NUM_BINS+1)];
		

    //Initializing host side global variables.
    cudaMemset(hough_space_device, 0, sizeof(int)*rho_max*(NUM_BINS+1));
    cudaMemset(eq_size_device, 0, sizeof(int));

    
    
    //Copy the Matrix to the edges_host to pass onto the GPU
    for( int i=0; i< h; i++) {
	for( int j=0; j< w; j++) {
		edges_host[i*w + j] = edges[i*w + j] ;
	}
    }

   

    error_code = cudaMemcpy((void*)edges_device, (void*)edges_host, sizeof(uchar)*h*w, cudaMemcpyHostToDevice);

    if(error_code !=0){
	cout<<"Copy of Edges Array from host to device Failed, Error code "<<error_code<<endl;
    }
	

    //Launching and Benchmarking the kernels.

    cudaEventRecord(start_cuda);
   
    voting_kernel <<<number_of_blocks_1, number_of_threads>>> ( edges_device, hough_space_device, w, h, diag);

    equation_forming_kernel<<<number_of_blocks_2, number_of_threads, sizeof(int)*(2*number_of_threads +1)>>> (hough_space_device, rho_max* (NUM_BINS+1), rho_arr_device, theta_arr_device, eq_size_device, line_threshold, diag) ;

    cudaEventRecord(stop_cuda);

    error_code = cudaEventSynchronize(stop_cuda);
 
    if(error_code !=0){
	cout<<"Kernel Launch Failed, Error code "<<error_code<<endl;
    }



    cudaEventElapsedTime(&ms, start_cuda, stop_cuda);


    //Copying the results back
    error_code = cudaMemcpy((void*)eq_size_host, (void*)eq_size_device, sizeof(int)*1, cudaMemcpyDeviceToHost);

    if(error_code !=0){
	cout<<"Copy of eq_size from device to host Failed, Error code "<<error_code<<endl;
    }


    error_code = cudaMemcpy((void*)rho_arr_host, (void*)rho_arr_device, sizeof(int)*rho_max*(NUM_BINS+1), cudaMemcpyDeviceToHost);

    if(error_code !=0){
	cout<<"Copy of rho_arr from device to host Failed, Error code "<<error_code<<endl;
    }


    error_code = cudaMemcpy((void*)theta_arr_host, (void*)theta_arr_device, sizeof(int)*rho_max*(NUM_BINS+1), cudaMemcpyDeviceToHost);

    if(error_code !=0){
	cout<<"Copy of theta_arr from device to host Failed, Error code "<<error_code<<endl;
    }

   cudaMemcpy((void*) hough_space_host, (void*)hough_space_device, sizeof(int)*rho_max*(NUM_BINS+1), cudaMemcpyDeviceToHost);
       
  
    //Freeing the memories
    cudaFree(edges_device);
    cudaFree(hough_space_device);
    cudaFree(rho_arr_device);
    cudaFree(theta_arr_device);

    cout<<"Time taken by GPU implementation  "<<ms<<endl;


    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////


 	

    ////////////////////////////////////////////////////////////
    ///////// Plotting Found Equations to the Image ////////////
    ////////////////////////////////////////////////////////////
      
    uchar* o_multi;


    //Making the monochrome image multi-color
    o_multi = new uchar [3*w*h];


    for(int i=0; i<h; i++) {
	for( int j=0; j<w; j++) {
		o_multi[3*(i*w+j)] = image_src[i*w + j];
		o_multi[3*(i*w+j)+1] = image_src[i*w + j];
		o_multi[3*(i*w+j)+2] = image_src[i*w + j];

	}
    }

    

    // Plotting lines from equations   
    //  For Host
    //  list<pair<int,int>>::iterator  it;
    //  for(int i = 0; i < equations_ref.size(); ++i) {
    //
    //   it = equations_ref.begin();

    //   advance(it, i);
    //   rho =  (*it).first;
    //   theta = (*it).second ;

    //   pair<int, int> p1, p2;  
    //   polarToCartesian(rho, theta, p1, p2);

    //   //cout<< p1 << ", " << p2 <<"\n";

    //   
    //    draw_line(o_multi, w,h, p1, p2);

    //}

    // For GPU
    
    //cout<<"Before Line Drawing"<<endl;
    for(int i=0; i< *eq_size_host; i++) {
        rho =  rho_arr_host[i];
        theta = theta_arr_host[i];
        pair<int, int> p1, p2;
        polarToCartesian(rho, theta, p1, p2);
        draw_line(o_multi,w,h, p1, p2);	
    }
  
      

    (void)stbi_write_jpg("output.jpg",  w, h, 3, o_multi, 100);

    	

    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////

    ////// Freeing up the memories/////////

    delete[] rho_arr_host;
    delete[] theta_arr_host;
    delete eq_size_host;
    delete[] edges_host;


    return 0;
}


void polarToCartesian(int rho, int theta, pair<int, int> &p1, pair<int, int> &p2) {


	double theta_rad = (((double)theta)/180) * PI;
	int x0 = round(rho * cos(theta_rad));
	int y0 = round(rho * sin(theta_rad));

	p1.first = round(x0 + 5000 * (-1*sin(theta_rad)));
	p1.second = round(y0 + 5000 * (cos(theta_rad)));

	p2.first = round(x0 - 5000 * (-1* sin(theta_rad)));
	p2.second = round(y0 - 5000 * (cos(theta_rad)));
}
