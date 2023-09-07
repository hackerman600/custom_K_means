#include <iostream>
#include <Eigen/Dense>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>

Eigen::MatrixXd matrix_creation(std::string path){
    std::ifstream x_train_file(path);
    std::string line;
    std::vector<std::vector<float>> matrix_2; 
    std::vector<float> matrx_2; 
    std::string val = "";
    int itter = 0;
    int com_itter = 0;
        

    while (std::getline(x_train_file, line)) {
        
        if (line[line.length() - 1] == ','){
            line = "," + line;
        } else {
            line = "," + line + ",";
        }
        
        com_itter = 0;
        itter = 0;

        for (char c : line){
            if (c == ','){
                com_itter += 1;
            }

            if (com_itter == 1 && c != ','){
                val += c;
            }

            if (com_itter == 2){
                matrx_2.push_back(std::stod(val));
                com_itter = 1;
                val = "";
            }

            itter += 1;

            if (itter == line.length()){
                matrix_2.push_back(matrx_2);
                matrx_2.clear();
            }
        }
                
    }

    //CONVERT INTO A MATRIX
    int rows = matrix_2.size();
    int cols = matrix_2[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            matrix(r,c) = matrix_2[r][c];
        }
    }

    return matrix; 

}


std::vector<int> return_clusters(Eigen::MatrixXd x_train, std::vector<int> centroids){
        std::vector<int> cluster_indexes_examples_belongto;
        for (int r = 0; r < x_train.rows(); r++){ //dp 0
             int smallest = 0;
             double smallest_absolute_difference = 0.0;
             for (int ind = 0; ind < centroids.size(); ind++){ //centroid 0
                 Eigen::MatrixXd centroid = x_train.row(centroids[ind]);
                 Eigen::MatrixXd datapoint = x_train.row(r);
                 double absolute_difference = std::sqrt(((centroid - datapoint).array() * (centroid - datapoint).array()).sum());
                  if (smallest_absolute_difference == 0.0){
                    smallest_absolute_difference = absolute_difference;
                 } else{
                    if (absolute_difference < smallest_absolute_difference){
                        smallest_absolute_difference = absolute_difference;
                        smallest = ind;
                    }
                 }
                                                
             } cluster_indexes_examples_belongto.push_back(smallest);  

        }

        return cluster_indexes_examples_belongto;
    

}

double return_average_median_of_absolute_differences(Eigen::MatrixXd x_train, std::vector<int> cluster_indexes, std::vector<int> centroids, int k){
    
    double median = 0.0;
    double total_median = 0.0;   

    std::vector<std::vector<double>> out = {};
    for (int i = 0; i < k; i++){
        std::vector<double> c;
        out.push_back(c);
    }
    
    for (int x_trainind = 0; x_trainind < x_train.rows(); x_trainind++){
        Eigen::MatrixXd dp = x_train.row(x_trainind);
        Eigen::MatrixXd centroid = x_train.row(centroids[cluster_indexes[x_trainind]]);
        double absolute_difference = std::sqrt(((centroid - dp).array() * (centroid - dp).array()).sum());
        out[cluster_indexes[x_trainind]].push_back(absolute_difference);
    }    //[1 2 3 4]

    for (int out_ind = 0; out_ind < out.size(); out_ind++){
        std::sort(out[out_ind].begin(),out[out_ind].end());
        int median_index = out[out_ind].size()/2;
        if (median_index % 2 == 0){
            median_index -= 1;
            out[out_ind].erase(out[out_ind].begin(), out[out_ind].begin()+median_index); 
            out[out_ind].erase(out[out_ind].end(), out[out_ind].end()-median_index); 
            median = (out[out_ind][0] + out[out_ind][1])/2;
            total_median += median;
        } 

        total_median += out[out_ind][median_index];
    
    }
    return total_median/k;

}


int main(){

    Eigen::MatrixXd x_train = matrix_creation("/home/kali/Desktop/machine_learning/neural_networks/from_scratch/cpp_networks/x_train.csv");
    std::cout << "x_train.size() = " << x_train.rows() << "," << x_train.cols() << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int>giveint(0,x_train.rows()-1);
    std::vector<double> model_comp;
    std::vector<std::vector<int>> best_ones;
    std::vector<std::vector<int>> count_clusters;

    int attempts = 10;
    int k = 6;
    for (int kay = 0; kay < k; kay++){
        std::vector<int> c;
        count_clusters.push_back(c);           
    }

    std::cout << "\nchoosing the model with the densest clusters \n\n";

    for (int att = 0; att < attempts; att++){
        std::vector<int> centroids;
        for (int kay = 0; kay < k; kay++){
            centroids.push_back(giveint(gen));
            
        }

        std::vector<int> cluster_indexes = return_clusters(x_train, centroids);
        best_ones.push_back(cluster_indexes);
        double average_median_of_absolute_differences = return_average_median_of_absolute_differences(x_train,cluster_indexes,centroids,k);
        std::cout << "average_median_of_absolute_differences = " << average_median_of_absolute_differences << std::endl;
        model_comp.push_back(average_median_of_absolute_differences);
    }

    int smallest_ind = 0;
    double prev_val = 0.0;
    for (int itt = 0; itt < model_comp.size(); itt++) {
        double cur_val = model_comp[itt];
        if (itt == 0){
           prev_val = cur_val; 
        } else {
           if (cur_val < prev_val){
                prev_val = cur_val;
                smallest_ind = itt;
           }
        }
    }

    std::cout << "\n\nmodel with the most dense clusters = " << model_comp[smallest_ind] << "\n\n" << std::endl; 
    for (int x = 0; x < k; x++){
        for (int c = 0; c < best_ones[smallest_ind].size(); c++){            
            if (best_ones[smallest_ind][c] == x){
                count_clusters[x].push_back(0);
            }
        }    
    }


    for (int c = 0; c < count_clusters.size(); c++){
        std::cout << "cluster[" << c << "] size = " << count_clusters[c].size() << std::endl;
    }


    return 0;
}