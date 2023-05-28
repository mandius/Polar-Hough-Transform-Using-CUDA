#!/usr/bin/env bash
#SBATCH --job-name=hough
#SBATCH --partition=wacc
#SBATCH --time=00-00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 -c 1
#SBATCH -o hough.out -e hough.err


#nv-nsight-cu-cli  --list-sets full -f -o hough.ncu-proj ./hough  sample_images/zebracrossing.jpg 160 100
 ./hough  sample_images/zebracrossing.jpg 160 100
# ./hough sample_images/texture.jpg 60 10
# ./hough sample_images/sudoku.jpg 150 70
# ./hough sample_images/Triangles.jpg 200 40

