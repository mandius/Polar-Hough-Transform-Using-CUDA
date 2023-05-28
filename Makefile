voting_in_shared_mem: hough_voting_in_shared_mem.cu
	nvcc hough_voting_in_shared_mem.cu  -g -G  -std=c++11 -Xcompiler -static-libstdc++ -o hough

baseline: hough_baseline.cu
	nvcc hough_baseline.cu -std=c++11   -Xcompiler -static-libstdc++  -o hough	



