#include<vector>

/*
 * Estimator Object class
 *
 * The estimator uses exponential smoothing.
 */

class Estimator{
    
    std::vector<double> estimates;
    double alpha;

public:
    Estimator(){
        this->alpha = 0.9999999999;
    }

    double Estimate(const double completed,const double duration){
        
        double lastEstimate = 0.0;
        
        if(estimates.size() != 0){
            lastEstimate = alpha * ((1 - completed) * duration) + (1-alpha) * estimates.at(estimates.size() - 1);
            estimates.push_back(lastEstimate);
        }
        else{
            lastEstimate = ((1 - completed) * duration);
            estimates.push_back(lastEstimate);
        }
        return lastEstimate;
    }
}; 