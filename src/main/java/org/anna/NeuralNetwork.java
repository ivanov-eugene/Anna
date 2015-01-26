package org.anna;

import java.util.*;

import org.jocl.*;

import static org.jocl.CL.*;

/**
 * Use this class to train and run an artificial neural network. The training algorithm uses the gradient decent 
 * algorithm. The class utilizes an OpenCL computing device (e.g. GPU) to achieve the best performance.
 * The class supports both single (32-bit) and double (64-bit) floating-point precision for math calculations.
 * Since a typical GPU has more 32-bit computing units than 64-bit ones, use single precision to achieve 
 * maximum performance. Use double precision when the computational precision becomes important for accurate training 
 * (e.g. a neural network with a large number of intermediate layers).
 * <p>
 * Here is a sample code demonstrating how to use this class:
 * <pre><code>
 * // Create a neural network that runs on OpenCL platform 0 and OpenCL device 0, 
 * // uses 64-bit floating point precision, takes 5 input features, has 3 layers with 4 nodes each
 * // and does a binary classification (i.e. has 2 labels)      
 * org.anna.NeuralNetwork nn = new org.anna.NeuralNetwork(0, 0, 
 *     new org.anna.NetworkDescription(org.anna.FloatingPointPrecision.DOUBLE, 5, 3, 4, 2));
 * 
 * // Train the neural network on the input samples and labels using 100
 * // as maximum number of iterations and no normalization
 * double[] samples = ... ; // the array length should be 5 (number of features) multiplied by the number of samples
 * int[] labels = ... ; // the array length should match the number of samples
 * nn.train(samples, labels, 100, 0.0, null);
 * 
 * // Classify a new sample by running the neural network on it and using 0.5 as the classification threshold
 * double[] newSample = ... ; // the array should contain 5 elements
 * double[] result = nn.run(newSample);
 * boolean classification = result[0] >= 0.5;                
 * </code></pre>
 * @author Eugene Ivanov
 */
public class NeuralNetwork {

    /**
     * Creates a new NeuralNetwork instance with the given NetworkDescription.
     * 
     * @param platformID A number starting with 0 identifying the OpenCL platform to use  
     * @param deviceID A number starting with 0 identifying the OpenCL device on the given OpenCL platform
     * @param networkDecription An object defining the neural network (should not be null)  
     */
    public NeuralNetwork(int platformID, int deviceID, NetworkDescription networkDecription) {
        if (networkDecription == null) 
            throw new IllegalArgumentException("networkDescription should not be null");
        this.networkDescription = networkDecription;
        initOpenCL(platformID, deviceID, networkDecription.precision);
    }
    
    /**
     * Use this method to retrieve the neural network description. Once trained, the object returned by
     * this method stores the training weights. Therefore, to persist a trained neural network, serialize
     * the NetworkDescription object returned by this method. To restore a trained neural network, 
     * deserialize the NetworkDescription object and pass it to NeuralNetwork constructor.  
     */
    public NetworkDescription getNetworkDescription() {
        return networkDescription;
    }

    NetworkDescription networkDescription;

    int numOfSamples;

    ArrayList<cl_mem> clmWeightsMatrices = null;
    ArrayList<cl_mem> clmActivationMatrices = null;

    cl_mem clmSamplesMatrix = null;
    cl_mem clmLabelsVector = null; 
    
    void initOpenCL(int platformIndex, int deviceIndex, FloatingPointPrecision precision) {
        computeDevice = ComputeDevice.getDeviceInstance(platformIndex, deviceIndex, precision);

        // Copy weights matrices to OpenCL device
        clmWeightsMatrices = new ArrayList<cl_mem>();
        for (int k = 0; k < networkDescription.numOfLayers; k++) {
            int width = networkDescription.weightsWidths.get(k);
            int height = networkDescription.weightsHeights.get(k);
            Object array = networkDescription.weightsArrays.get(k);

            cl_mem clmWeightsMatrix = clCreateBuffer(computeDevice.getContext(), CL_MEM_READ_WRITE,
                    computeDevice.bufferSize(height * width), null, null);
            clmWeightsMatrices.add(clmWeightsMatrix);
            if (precision.isDouble())
                clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmWeightsMatrix, CL_TRUE, 0, 
                        computeDevice.bufferSize(height * width), Pointer.to((double[])array), 0, null, null);
            else
                clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmWeightsMatrix, CL_TRUE, 0, 
                        computeDevice.bufferSize(height * width), Pointer.to((float[])array), 0, null, null);
        }
    }

    // Null indicates that the NeuralNetwork instance was released or is not initialized yet
    ComputeDevice computeDevice = null;

    void confirmActiveState() {
        if (computeDevice == null) 
            throw new IllegalStateException("The NeuralNetwork instance was released or not initialized yet");
    }

    static float[] doubleArrayToFloatArray(double[] doubleArray) {
        float floatArray[] = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++)
            floatArray[i] = (float) doubleArray[i];
        return floatArray;
    }

    static double[] floatArrayToDoubleArray(float[] floatArray) {
        double doubleArray[] = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++)
            doubleArray[i] = floatArray[i];
        return doubleArray;
    }

    void internalSetInputMatrix(FloatingPointPrecision inputArrayPrecision, Object inputMatrix) {
        // Check input
        int samplesMatrixSize = inputArrayPrecision.isDouble() ? 
                ((double[])inputMatrix).length : ((float[])inputMatrix).length;
        if (samplesMatrixSize % networkDescription.numOfFeatures != 0)
            throw new IllegalArgumentException("The size of input matrix should be a mutiple of number of features");
        
        // Release OpenCL buffers if it are already allocated to prevent a memory leak  
        if (clmSamplesMatrix != null) {
            clReleaseMemObject(clmSamplesMatrix);
            clmSamplesMatrix = null;
        }
        if (clmActivationMatrices != null) {
            for (int k = 0; k < networkDescription.numOfLayers; k++) {
                clReleaseMemObject(clmActivationMatrices.get(k));
            }
        }
        
        // Convert input array if necessary 
        Object inputArray = null;
        if (inputArrayPrecision == networkDescription.precision)
            inputArray = inputMatrix;
        else if (inputArrayPrecision.isDouble() && !networkDescription.precision.isDouble())
            inputArray = doubleArrayToFloatArray((double[])inputMatrix);
        else if (!inputArrayPrecision.isDouble() && networkDescription.precision.isDouble())
            inputArray = floatArrayToDoubleArray((float[])inputMatrix);

        // Copy samples matrix to OpenCL device
        numOfSamples = samplesMatrixSize / networkDescription.numOfFeatures; 
        clmSamplesMatrix = clCreateBuffer(computeDevice.getContext(), CL_MEM_READ_ONLY,
                computeDevice.bufferSize(samplesMatrixSize), null, null);
        if (networkDescription.precision.isDouble())
            clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmSamplesMatrix, CL_TRUE, 0, 
                    computeDevice.bufferSize(samplesMatrixSize), Pointer.to((double[])inputArray), 0, null, null);
        else
            clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmSamplesMatrix, CL_TRUE, 0, 
                    computeDevice.bufferSize(samplesMatrixSize), Pointer.to((float[])inputArray), 0, null, null);

        // Create activation matrices at OpenCL device
        clmActivationMatrices = new ArrayList<cl_mem>();
        for (int k = 0; k < networkDescription.numOfLayers; k++) {
            int width = networkDescription.weightsWidths.get(k);
            clmActivationMatrices.add(clCreateBuffer(computeDevice.getContext(), CL_MEM_READ_WRITE, 
                    computeDevice.bufferSize(numOfSamples * width), null, null));
        }
    }
    
    void internalSetLabelsVector(int[] labelsVector) {
        // Release the OpenCL buffer if it is already allocated to prevent a memory leak  
        if (clmLabelsVector != null) {
            clReleaseMemObject(clmLabelsVector);
            clmLabelsVector = null;
        }

        // Copy labels vector to OpenCL device
        if (labelsVector != null) {
            clmLabelsVector = clCreateBuffer(computeDevice.getContext(), CL_MEM_READ_ONLY,
                    Sizeof.cl_int * labelsVector.length, null, null);
            clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmLabelsVector, CL_TRUE, 0, 
                    Sizeof.cl_int * labelsVector.length, Pointer.to(labelsVector), 0, null, null);
        }
    }

    /**
     * Release all OpenCL artifacts. The NeuralNetwork instance is no longer usable after calling 
     * this method. Calling the method after the instance has already been released does nothing.
     */
    public void release() {
        // Do nothing if the instance is already released
        if (computeDevice == null) return; 

        if (clmWeightsMatrices != null) {
            for (int k = 0; k < networkDescription.numOfLayers; k++) {
                clReleaseMemObject(clmWeightsMatrices.get(k));
            }
            clmWeightsMatrices = null;
        }
        if (clmActivationMatrices != null) {
            for (int k = 0; k < networkDescription.numOfLayers; k++) {
                clReleaseMemObject(clmActivationMatrices.get(k));
            }
            clmActivationMatrices = null;
        }
        if (clmSamplesMatrix != null) {
            clReleaseMemObject(clmSamplesMatrix);
            clmSamplesMatrix = null;
        }
        if (clmLabelsVector != null) {
            clReleaseMemObject(clmLabelsVector);
            clmLabelsVector = null;
        }
        if (computeDevice != null) {
            computeDevice.release();
            computeDevice = null;
        }
    }

    /**
     *  This method will make sure that all OpenCL artifacts are released when the NeuralNetwork instance 
     *  is garbage collected. 
     */
    protected void finalize() throws Throwable {
        try {
            release();
        } finally {
            super.finalize();
        }
    }

    /**
     * Run the neural network on the input sample provided with a double array. The input array must be  
     * a row-by-row packed matrix. Each input matrix row should contain numOfFeatures (see 
     * {@link NetworkDescription} class) elements and represents one input sample. The input matrix should 
     * contain at least one sample. If the neural network was created with single floating point precision, 
     * the input array is automatically down-converted to a float array which causes a slight performance penalty 
     * and loss of precision.
     * <p>
     * The method returns a double array which represents a row-by-row packed matrix. Each output matrix row
     * contains numOfLabelCategories (see {@link NetworkDescription} class) elements. Each output matrix row
     * corresponds to a row in the input matrix. Each output matrix item is a floating point number between 0 and 1.
     * The number indicates a certainty with which the corresponding input sample belongs to the category
     * indexed by column number (the first column is indexed with 0 and corresponds to the first label category).
     * <p>     
     * Calling this method on an untrained neural network produces an array of random numbers.  
     */
    public double[] run(double[] inputMatrix) {
        internalRun(FloatingPointPrecision.DOUBLE, inputMatrix);
        return (double[]) getResultMatrix(true);
    }

    /**
     * Run the neural network on the input sample provided with a float array. The input array must be  
     * a row-by-row packed matrix. Each input matrix row should contain numOfFeatures (see 
     * {@link NetworkDescription} class) elements and represents one input sample. The input matrix should 
     * contain at least one sample. If the neural network was created with double floating point precision, 
     * the input array is automatically up-converted to a double array which causes a slight performance penalty.
     * <p>
     * The method returns a float array which represents a row-by-row packed matrix. Each output matrix row
     * contains numOfLabelCategories (see {@link NetworkDescription} class) elements. Each output matrix row
     * corresponds to a row in the input matrix. Each output matrix item is a floating point number between 0 and 1.
     * The number indicates a certainty with which the corresponding input sample belongs to the category
     * indexed by column number (the first column is indexed with 0 and corresponds to the first label category).
     * <p>     
     * Calling this method on an untrained neural network produces an array of random numbers.  
     */
    public float[] run(float[] inputMatrix) {
        internalRun(FloatingPointPrecision.SINGLE, inputMatrix);
        return (float[]) getResultMatrix(false);
    }

    void internalRun(FloatingPointPrecision inputArrayPrecision, Object inputMatrix) {
        // Check state and input
        confirmActiveState();
        if (inputMatrix == null)
            throw new IllegalArgumentException("The input matrix should not be null");
        
        internalSetInputMatrix(inputArrayPrecision, inputMatrix);
        forwardPropagation();
    }
    
    // Returns double[] or float[] depending on the returnDoubleArray parameter
    Object getResultMatrix(boolean returnDoubleArray) {
        confirmActiveState();
        if (clmActivationMatrices == null || clmActivationMatrices.get(clmActivationMatrices.size() - 1) == null)
            throw new IllegalStateException("The NeuralNetwork has not been run yet");
        
        int size = networkDescription.numOfLabelCategories * numOfSamples;
        if (networkDescription.precision.isDouble()) {
            double[] resultMatrix = new double[size];
            clEnqueueReadBuffer(computeDevice.getCommandQueue(), 
                    clmActivationMatrices.get(clmActivationMatrices.size() - 1), CL_TRUE, 0,
                    computeDevice.bufferSize(size), Pointer.to(resultMatrix), 0, null, null);
            if (returnDoubleArray)
                return resultMatrix;
            else 
                return doubleArrayToFloatArray(resultMatrix); 
        }
        else {
            float[] resultMatrix = new float[size];
            clEnqueueReadBuffer(computeDevice.getCommandQueue(), 
                    clmActivationMatrices.get(clmActivationMatrices.size() - 1), CL_TRUE, 0,
                    computeDevice.bufferSize(size), Pointer.to(resultMatrix), 0, null, null);
            if (!returnDoubleArray)
                return resultMatrix;
            else 
                return floatArrayToDoubleArray(resultMatrix); 
        }
    }

    // Execute the forward propagation algorithm on each layer
    void forwardPropagation() {
        long startTime = System.currentTimeMillis();

        for (int k = 0; k < networkDescription.weightsArrays.size(); k++) {
            if (k == 0) {
                // Very first layer
                computeDevice.forwardPropagation(
                        clmSamplesMatrix, clmWeightsMatrices.get(k), clmActivationMatrices.get(k), 
                        (networkDescription.weightsHeights.get(k) - 1) * networkDescription.weightsWidths.get(k), 
                        networkDescription.numOfFeatures, numOfSamples, networkDescription.weightsWidths.get(k));
            }
            else {
                computeDevice.forwardPropagation(
                        clmActivationMatrices.get(k - 1), clmWeightsMatrices.get(k), clmActivationMatrices.get(k), 
                        (networkDescription.weightsHeights.get(k) - 1) * networkDescription.weightsWidths.get(k), 
                        networkDescription.weightsWidths.get(k - 1), numOfSamples, networkDescription.weightsWidths.get(k));
            }
        }

        clFinish(computeDevice.getCommandQueue());
        timeInOpenCL += System.currentTimeMillis() - startTime;
    }

    long timeInOpenCL = 0;

    // Calculate cost function with given regularization parameter lambda
    double calculateCost(double lambda) {
        confirmActiveState();
        long startTime = System.currentTimeMillis();

        cl_mem clmLastMatrixA = clmActivationMatrices.get(clmActivationMatrices.size() - 1);
        int width = networkDescription.weightsWidths.get(networkDescription.weightsWidths.size() - 1);

        cl_mem clmLogMatrix = clCreateBuffer(
                computeDevice.getContext(), CL_MEM_READ_WRITE, computeDevice.bufferSize(numOfSamples * width), null, null);
        computeDevice.calculateLogPartCostFunction(
                clmLabelsVector, clmLastMatrixA, clmLogMatrix, numOfSamples, width);
        double costSum = computeDevice.calculateTotal(clmLogMatrix, numOfSamples * width);
        clReleaseMemObject(clmLogMatrix);
        costSum = costSum / numOfSamples;

        // Add regularization
        double regSum = 0;
        if (lambda != 0.0) {
            for (int i = 0; i < networkDescription.weightsArrays.size(); i++) {
                regSum += computeDevice.calculateSquaredTotal(clmWeightsMatrices.get(i), 
                        networkDescription.weightsWidths.get(i) * 
                        (networkDescription.weightsHeights.get(i) - 1)); // exclude bias row 
            }
            regSum = (regSum / (numOfSamples * 2)) * lambda;
        }

        clFinish(computeDevice.getCommandQueue());
        timeInOpenCL += System.currentTimeMillis() - startTime;

        return costSum + regSum;
    }

    // Calculate the gradient with given regularization parameter lambda.
    // Caller is responsible for releasing the result OpenCL buffer.
    LinkedList<cl_mem> calculateGradient(double lambda) {
        confirmActiveState();
        long startTime = System.currentTimeMillis();

        cl_mem clmPriorDeltaMatrix = null;
        LinkedList<cl_mem> result = new LinkedList<cl_mem>();
        for (int weightsIndex = clmWeightsMatrices.size() - 1; weightsIndex >= 0 ; weightsIndex--) {
            if (weightsIndex == clmWeightsMatrices.size() - 1) {
                cl_mem clmActivationsMatrix = clmActivationMatrices.get(weightsIndex);
                cl_mem clmPriorActivationsMatrix = clmActivationMatrices.get(weightsIndex - 1);
                cl_mem clmWeightsMatrix = clmWeightsMatrices.get(weightsIndex);
                int height = networkDescription.weightsHeights.get(weightsIndex);
                int width = networkDescription.weightsWidths.get(weightsIndex);
                cl_mem clmResultMatrix = clCreateBuffer(computeDevice.getContext(), 
                        CL_MEM_READ_WRITE, computeDevice.bufferSize(height * width), null, null);
                computeDevice.calculateGradient(
                        clmActivationsMatrix, clmPriorActivationsMatrix, clmLabelsVector, clmWeightsMatrix, 
                        clmResultMatrix, height, width, numOfSamples, lambda);
                result.add(clmResultMatrix);
            } 
            else {
                cl_mem clmActivationsMatrix = clmActivationMatrices.get(weightsIndex);
                cl_mem clmPriorActivationsMatrix = weightsIndex == 0 ? 
                        clmSamplesMatrix : clmActivationMatrices.get(weightsIndex - 1);
                cl_mem clmWeightsMatrix = clmWeightsMatrices.get(weightsIndex);
                int height = networkDescription.weightsHeights.get(weightsIndex);
                int width = networkDescription.weightsWidths.get(weightsIndex);
                cl_mem clmNextWeightsMatrix = clmWeightsMatrices.get(weightsIndex + 1);
                int nextWidth = networkDescription.weightsWidths.get(weightsIndex + 1);

                cl_mem clmCurrentDeltaMatrix = clmPriorDeltaMatrix;
                cl_mem clmCurrentLabelsVector = null; 
                if (weightsIndex == clmWeightsMatrices.size() - 2) { 
                    clmCurrentDeltaMatrix = clmActivationMatrices.get(clmActivationMatrices.size() - 1);
                    clmCurrentLabelsVector = clmLabelsVector;
                }

                cl_mem clmMatrixDelta = clCreateBuffer(computeDevice.getContext(), 
                        CL_MEM_READ_WRITE, computeDevice.bufferSize(numOfSamples * width), null, null);
                computeDevice.calculateGradientDelta(clmCurrentDeltaMatrix, clmCurrentLabelsVector, 
                        clmNextWeightsMatrix, clmActivationsMatrix, clmMatrixDelta, nextWidth, width, numOfSamples);

                cl_mem clmResultMatrix = clCreateBuffer(computeDevice.getContext(), 
                        CL_MEM_READ_WRITE, computeDevice.bufferSize(height * width), null, null);
                computeDevice.calculateGradient(
                        clmMatrixDelta, clmPriorActivationsMatrix, null, clmWeightsMatrix, 
                        clmResultMatrix, height, width, numOfSamples, lambda);

                if (clmPriorDeltaMatrix != null) {
                    clReleaseMemObject(clmPriorDeltaMatrix);
                }
                clmPriorDeltaMatrix = clmMatrixDelta;

                result.addFirst(clmResultMatrix);
            }
        }

        if (clmPriorDeltaMatrix != null) {
            clReleaseMemObject(clmPriorDeltaMatrix);
        }

        clFinish(computeDevice.getCommandQueue());
        timeInOpenCL += System.currentTimeMillis() - startTime;

        return result;
    }
    
    // Used for testing only
    void setLablesVector(int[] labelsVector) {
        if (clmLabelsVector != null) {
            clReleaseMemObject(clmLabelsVector);
            clmLabelsVector = null;
        }

        clmLabelsVector = clCreateBuffer(computeDevice.getContext(), CL_MEM_READ_ONLY,
                Sizeof.cl_int * labelsVector.length, null, null);
        clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmLabelsVector, CL_TRUE, 0, 
                Sizeof.cl_int * labelsVector.length, Pointer.to(labelsVector), 0, null, null);
    }

    // Break up the input double[] into pieces and assign each piece to a weights matrix
    void arrayToWeights(double[] array) {
        int offset = 0;
        for (int i = 0; i < networkDescription.weightsArrays.size(); i++) {
            int size = networkDescription.weightsWidths.get(i) * networkDescription.weightsHeights.get(i);
            if (networkDescription.precision.isDouble()) {
                System.arraycopy(array, offset, networkDescription.weightsArrays.get(i), 0, size);
            }
            else {
                // Convert double[] to float[]
                for (int j = 0; j < size; j++) 
                    ((float[])networkDescription.weightsArrays.get(i))[j] = (float)array[offset + j];
            }
            offset += size;

            // Copy weights to OpenCL device
            int width = networkDescription.weightsWidths.get(i);
            int height = networkDescription.weightsHeights.get(i);
            cl_mem clmWeightsMatrix = clCreateBuffer(computeDevice.getContext(), 
                    CL_MEM_READ_WRITE, computeDevice.bufferSize(height * width), null, null);
            Pointer pointer = networkDescription.precision.isDouble() ? 
                        Pointer.to((double[])networkDescription.weightsArrays.get(i)) :
                        Pointer.to((float[])networkDescription.weightsArrays.get(i));
            clEnqueueWriteBuffer(computeDevice.getCommandQueue(), clmWeightsMatrix, CL_TRUE, 0, 
                    computeDevice.bufferSize(height * width), pointer, 0, null, null);
            clReleaseMemObject(clmWeightsMatrices.get(i));
            clmWeightsMatrices.set(i, clmWeightsMatrix);
        }
        
        forwardPropagation();
    }

    // Merge all weights matrices into one double[]
    double[] weightsToArray() {
        int length = 0;
        for (int i = 0; i < networkDescription.weightsArrays.size(); i++) {
            length += networkDescription.weightsWidths.get(i) * networkDescription.weightsHeights.get(i);
        }
        double[] result = new double[length];
        int offset = 0;
        for (int i = 0; i < networkDescription.weightsArrays.size(); i++) {
            int size = networkDescription.weightsWidths.get(i) * networkDescription.weightsHeights.get(i);
            if (networkDescription.precision.isDouble()) {
                System.arraycopy(networkDescription.weightsArrays.get(i), 0, result, offset, size);
            }
            else {
                // convert float[] to double[]
                for (int j = 0; j < size; j++) 
                    result[offset + j] = ((float[])networkDescription.weightsArrays.get(i))[j];
            }
            offset += size;
        }
        return result;
    }

    // Merge all input buffers stored on the OpenCL device into one double[]
    double[] gradientsToArray(LinkedList<cl_mem> list) {
        int length = 0;
        for (int i = 0; i < list.size(); i++) {
            length += networkDescription.weightsWidths.get(i) * networkDescription.weightsHeights.get(i);
        }
        double[] result = new double[length];
        int offset = 0;
        for (int i = 0; i < list.size(); i++) {
            int size = networkDescription.weightsWidths.get(i) * networkDescription.weightsHeights.get(i);
            if (networkDescription.precision.isDouble()) {
                clEnqueueReadBuffer(computeDevice.getCommandQueue(), list.get(i), CL_TRUE, 0,
                        computeDevice.bufferSize(size), Pointer.to(result).withByteOffset(offset), 0, null, null);
                offset += computeDevice.bufferSize(size);
            }
            else {
                // Convert float[] to double[]
                float[] temp = new float[size];
                clEnqueueReadBuffer(computeDevice.getCommandQueue(), list.get(i), CL_TRUE, 0,
                        computeDevice.bufferSize(size), Pointer.to(temp), 0, null, null);
                for (int j = 0; j < size; j++) 
                    result[offset + j] = temp[j];
                offset += size;
            }
            clReleaseMemObject(list.get(i));
        }
        return result;
    }

    /**
     * Train the neural network on given samples and labels. 
     * <p>
     * The samplesMatrix array specifies a row-by-row packed matrix. Each matrix row should contain numOfFeatures 
     * (see {@link NetworkDescription} class) elements and represents one input sample. The input matrix should 
     * contain at least one sample. If the neural network was created with single floating point precision, 
     * the samples array is automatically converted to a float array which causes a slight performance penalty and 
     * a potential loss of accuracy.
     * <p>
     * Each item in the labelsVector array specifies a classification label (a number 0 through numOfLabelCategories-1) 
     * for each sample in the sampleMatrix array. 
     * <p>
     * The maxNumOfIterations parameter defines the maximum number of iterations used by the training algorithm. 
     * To make sure the training algorithm completes within a reasonable time, do not set this number larger than
     * a couple of hundreds.
     * <p>
     * The lambda parameter specifies the amount of regularization used by the training algorithm. Set it to 0,
     * to avoid regularization.
     * <p>
     * To monitor the training progress, implement the {@link TrainingProgress} interface and pass an instance via
     * the progress parameter. Set this parameter to null, if the training progress does not need to be tracked.
     */
    public void train(double[] samplesMatrix, int[] labelsVector, 
            int maxNumOfIterations, double lambda, TrainingProgress progress) {
        internalTrain(FloatingPointPrecision.DOUBLE, samplesMatrix, labelsVector, maxNumOfIterations, lambda, progress);
    }

    /**
     * Train the neural network on given samples and labels. 
     * <p>
     * The samplesMatrix array specifies a row-by-row packed matrix. Each matrix row should contain numOfFeatures 
     * (see {@link NetworkDescription} class) elements and represents one input sample. The input matrix should 
     * contain at least one sample. If the neural network was created with double floating point precision, 
     * the samples array is automatically converted to a double array which causes a slight performance penalty.
     * <p>
     * Each item in the labelsVector array specifies a classification label (a number 0 through numOfLabelCategories-1) 
     * for each sample in the sampleMatrix array. 
     * <p>
     * The maxNumOfIterations parameter defines the maximum number of iterations used by the training algorithm. 
     * To make sure the training algorithm completes within a reasonable time, do not set this number larger than
     * a couple of hundreds.
     * <p>
     * The lambda parameter specifies the amount of regularization used by the training algorithm. Set it to 0,
     * to avoid regularization.
     * <p>
     * To monitor the training progress, implement the {@link TrainingProgress} interface and pass an instance via
     * the progress parameter. Set this parameter to null, if the training progress does not need to be tracked.
     */
    public void train(float[] samplesMatrix, int[] labelsVector, 
            int maxNumOfIterations, double lambda, TrainingProgress progress) {
        internalTrain(FloatingPointPrecision.SINGLE, samplesMatrix, labelsVector, maxNumOfIterations, lambda, progress);
    }

    void internalTrain(FloatingPointPrecision inputArrayPrecision, Object inputMatrix, int[] labelsVector, 
            int maxNumOfIterations, double lambda, TrainingProgress progress) {
        // check state and input
        confirmActiveState();
        if (labelsVector == null)
            throw new IllegalArgumentException("The labelsVector should not be null");
        
        internalSetLabelsVector(labelsVector);
        
        internalRun(inputArrayPrecision, inputMatrix);

        GradientDescentMinimizer minimizer = new GradientDescentMinimizer() {
            double lambda;
            TrainingProgress progress;
            
            GradientDescentMinimizer init(double lambda, TrainingProgress progress) {
                this.lambda = lambda;
                this.progress = progress;
                return this;
            }

            @Override
            public double[] getWeights() {
                return weightsToArray();
            }

            @Override
            public void setWeights(double[] array) {
                arrayToWeights(array);
            }

            @Override
            public double[] advanceGradientDescent() {
                return gradientsToArray(calculateGradient(lambda));
            }

            @Override
            public double costFunction() {
                return calculateCost(lambda);
            }

            @Override
            public void reportProgress(int iteration, double cost) {
                if (progress != null)
                    progress.reportProgress(iteration, cost);
            }
        }.init(lambda, progress);

        minimizer.minimize(maxNumOfIterations);
    }

}
