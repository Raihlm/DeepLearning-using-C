#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.0001
#define EPOCHS 10000


typedef struct{
    // Layer 1: input to hidden
    double weights_input_hidden[INPUT_SIZE][HIDDEN_SIZE]; // 4 x 3 matrix
    double bias_hidden[HIDDEN_SIZE]; // 4 x 1 vector
    
    // Layer 2: hidden to output
    double weights_hidden_output[HIDDEN_SIZE][OUTPUT_SIZE];
    double bias_output[OUTPUT_SIZE];

    // Activations(saved for backpropagation)
    double hidden_[HIDDEN_SIZE]; // after activation
    double output[OUTPUT_SIZE]; // final result

    // Layer 4: 
    double z_hidden[HIDDEN_SIZE]; // before activation(ReLU)
    double z_output[OUTPUT_SIZE]; // before output

} NeuralNetwork;


// relu and relu_derivative are for actvivation function
double relu(double x){
    return x > 0? x : 0;
}

double relu_derivative(double x){
    return x > 0? 1.0 : 0.0;
}

void init_network(NeuralNetwork *nn){
    srand(time(NULL)); // Seed the random number generator

    // initialize input -> hidden layer weights and biases
    for(int i =0;i < HIDDEN_SIZE; i++){
        for(int j = 0; j < INPUT_SIZE; j++){
            //random weights between -0.5 and 0.5
            nn->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }

    for(int i = 0; i < OUTPUT_SIZE; i++){
        for(int j = 0; j < HIDDEN_SIZE;j++){
            nn->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
}

void forward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE]){

    // input -> hidden
    for (int i = 0; i < HIDDEN_SIZE; i++){
        // this is z = wx + b
        nn->z_hidden[i] = nn->bias_hidden[i];
        for(int j =0;j < INPUT_SIZE;j++){
            nn->z_hidden[i] += nn->weights_input_hidden[j][i] * input[j];
        }

        nn->hidden_[i] = relu(nn->z_hidden[i]);
    }

    // hidden -> output
    for(int i = 0; i < OUTPUT_SIZE;i++){
        nn->z_output[i] = nn->bias_output[i];
        for(int j = 0;j < HIDDEN_SIZE; j++){
            nn->z_output[i] += nn->weights_hidden_output[i][j] * nn->hidden_[j];
        }
        nn->output[i] = nn->z_output[i]; // for regression, no activation
    }


}

void backward_propagation(NeuralNetwork *nn, double input[INPUT_SIZE],double target[OUTPUT_SIZE]){
    //1. calculate output layer error
    // mse equation: error = predicted - actual
    double output_error[OUTPUT_SIZE];
    for(int i = 0; i < OUTPUT_SIZE; i++){
        output_error[i] = nn->output[i] - target[i];
    }

    //2. backpropogate to hidden layer
    // hidden-error = output_error x weight x ReLU'(z)
    double hidden_error[HIDDEN_SIZE];

    for (int i = 0; i < HIDDEN_SIZE; i++){

        hidden_error[i]= 0.0;
        for (int j = 0; j< OUTPUT_SIZE; j++){
            hidden_error[i] += output_error[j] * nn->weights_hidden_output[j][i];
        }
        hidden_error[i] *= relu_derivative(nn->z_hidden[i]);
    }

    //3. update hidden -> output weights
    for (int i = 0; i < OUTPUT_SIZE;i++){
        for(int j = 0; j < HIDDEN_SIZE;j++){
            nn->weights_hidden_output[i][j] -= LEARNING_RATE * output_error[i] * nn->hidden_[j];
        }
        nn->bias_output[i] -= LEARNING_RATE * output_error[i];
    }

    // STEP 4: Update input → hidden weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->weights_input_hidden[i][j] -= LEARNING_RATE * hidden_error[i] * input[j];
        }
        nn->bias_hidden[i] -= LEARNING_RATE * hidden_error[i];
    }

}

void normalize_data(double data[][INPUT_SIZE], int num_samples,
                   double min[INPUT_SIZE], double max[INPUT_SIZE]) {
    // Find min and max for each feature
    for (int j = 0; j < INPUT_SIZE; j++) {
        min[j] = data[0][j];
        max[j] = data[0][j];
        for (int i = 1; i < num_samples; i++) {
            if (data[i][j] < min[j]) min[j] = data[i][j];
            if (data[i][j] > max[j]) max[j] = data[i][j];
        }
    }
    
    // Normalize: (value - min) / (max - min)
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            data[i][j] = (data[i][j] - min[j]) / (max[j] - min[j]);
        }
    }
}

int main(){
    NeuralNetwork nn;
    init_network(&nn);

    // Example training data (XOR problem)
    double inputs[4][INPUT_SIZE] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double targets[4][OUTPUT_SIZE] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Normalize input data
    double min[INPUT_SIZE], max[INPUT_SIZE];
    normalize_data(inputs, 4, min, max);

    // Train the network
    for (int epoch = 0; epoch < 10000; epoch++) {
        for (int i = 0; i < 4; i++) {
            forward_propagation(&nn, inputs[i]);
            backward_propagation(&nn, inputs[i], targets[i]);
        }
    }

    // Test the network
    for (int i = 0; i < 4; i++) {
        forward_propagation(&nn, inputs[i]);
        printf("Input: [%f, %f] - Predicted: %f\n", inputs[i][0], inputs[i][1], nn.output[0]);
    }

    return 0;
}