package org.anna;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;

import org.jocl.Pointer;
import org.jocl.cl_mem;

import static org.jocl.CL.*;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class TestNeuralNetwork {
    int platformID;
    int deviceID;

    int numOfSamples = 5000;
    int numOfFeatures = 400;
    int numOfNodesInLayer = 25;
    int numOfLayers = 2;
    int numOfLabelCategories = 10;

    ArrayList<double[]> weights;
    double[] samples;
    int[] labels;

    private String propPlatformID = "anna.platformID";
    private String propDeviceID = "anna.deviceID";

    private static float[] doubleArrayToFloatArray(double[] doubleArray) {
        float floatArray[] = new float[doubleArray.length];
        for (int i = 0; i < doubleArray.length; i++)
            floatArray[i] = (float) doubleArray[i];
        return floatArray;
    }

    private static ArrayList<float[]> listDoubleArrayToListFloatArray(ArrayList<double[]> listDoubleArray) {
        ArrayList<float[]> listFloatArray = new ArrayList<float[]>(listDoubleArray.size());
        for (double[] a : listDoubleArray) 
            listFloatArray.add(doubleArrayToFloatArray(a));
        return listFloatArray;
    }

    @Before
    public void loadData() throws IOException {
        platformID = Integer.valueOf(System.getProperty(propPlatformID, "0"));
        deviceID = Integer.valueOf(System.getProperty(propDeviceID, "0"));

        // Load first weights matrix
        double[] weightsArray1 = new double[(numOfFeatures + 1) * numOfNodesInLayer]; //10025
        BufferedReader weightsFile1 = new BufferedReader(new InputStreamReader(
                getClass().getClassLoader().getResourceAsStream("Weights1.txt")));
        String line = weightsFile1.readLine();
        int i = 0;
        while (line != null) {
            weightsArray1[i] = Double.parseDouble(line);
            i++;
            line = weightsFile1.readLine();
        }
        weightsFile1.close();

        // Load second weights matrix
        double[] weightsArray2 = new double[(numOfNodesInLayer + 1) * numOfLabelCategories]; //260
        BufferedReader weightsFile2 = new BufferedReader(new InputStreamReader(
                getClass().getClassLoader().getResourceAsStream("Weights2.txt")));
        line = weightsFile2.readLine();
        i = 0;
        while (line != null) {
            weightsArray2[i] = Double.parseDouble(line);
            i++;
            line = weightsFile2.readLine();
        }
        weightsFile2.close();

        weights = new ArrayList<double[]>();
        weights.add(weightsArray1);
        weights.add(weightsArray2);

        // Load training samples (saved in binary format to save space)
        samples = new double[numOfSamples*numOfFeatures]; //2000000 
        DataInputStream fileX = new DataInputStream(new BufferedInputStream(
                getClass().getClassLoader().getResourceAsStream("Samples.bin")));
        for (int k = 0; k < samples.length; k++) {
            samples[k] = fileX.readDouble();
        }
        fileX.close();

        // Load training labels 
        labels = new int[numOfSamples];
        BufferedReader fileY = new BufferedReader(new InputStreamReader(
                getClass().getClassLoader().getResourceAsStream("Labels.txt")));
        line = fileY.readLine();
        i = 0;
        while (line != null) {
            labels[i] = Integer.parseInt(line);
            i++;
            line = fileY.readLine();
        }
        fileY.close();
    }

    private void internalTestCostFunctionNoNormalization(NeuralNetwork nn, boolean useDoublePrecision, Object samples) {
        if (useDoublePrecision)
            nn.run((double[])samples);
        else
            nn.run((float[])samples);
        nn.setLablesVector(labels);
        double cost = nn.calculateCost(0.0f);
        System.out.println(cost);
        assertTrue(Math.abs(cost - 0.287629) < 0.000001);
        nn.release();
    }

    private void internalTestCostFunctionWithNormalization(NeuralNetwork nn, boolean useDoublePrecision, Object samples) {
        if (useDoublePrecision)
            nn.run((double[])samples);
        else
            nn.run((float[])samples);
        nn.setLablesVector(labels);
        double cost = nn.calculateCost(1.0f);
        System.out.println(cost);
        assertTrue(Math.abs(cost - 0.383770) < 0.000001);
        nn.release();
    }
    
    @Test
    public void testCostFunctionWithSinglePrecissionNoNormalization() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.SINGLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = listDoubleArrayToListFloatArray(weights);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestCostFunctionNoNormalization(nn, false, doubleArrayToFloatArray(samples));
    }

    @Test
    public void testCostFunctionWithSinglePrecissionWithNormalization() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.SINGLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = listDoubleArrayToListFloatArray(weights);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestCostFunctionWithNormalization(nn, false, doubleArrayToFloatArray(samples));
    }

    @Test
    public void testCostFunctionWithDoublePrecissionNoNormalization() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = weights;
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestCostFunctionNoNormalization(nn, true, samples);
    }

    @Test
    public void testCostFunctionWithDoublePrecissionWithNormalization() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = weights;
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestCostFunctionWithNormalization(nn, true, samples);
    }
    
    @Test
    public void testDoublePrecisonTopologySerialization() throws IOException, ClassNotFoundException {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = weights;
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);

        File tempFile = File.createTempFile("anna",".nnd");
        tempFile.deleteOnExit();
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(tempFile, false));
        oos.writeObject(nn.getNetworkDescription());
        oos.close();
        nn.release();

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(tempFile));
        nnd = (NetworkDescription) ois.readObject();
        ois.close();
        nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestCostFunctionNoNormalization(nn, true, samples);
    }
    
    @Test
    public void testSinglePrecisonTopologySerialization() throws IOException, ClassNotFoundException {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.SINGLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = listDoubleArrayToListFloatArray(weights);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);

        File tempFile = File.createTempFile("anna",".nnd");
        tempFile.deleteOnExit();
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(tempFile, false));
        oos.writeObject(nn.getNetworkDescription());
        oos.close();
        nn.release();

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(tempFile));
        nnd = (NetworkDescription) ois.readObject();
        ois.close();
        nn = new NeuralNetwork(platformID, deviceID, nnd);
        float[] singlePrecisionSamples = doubleArrayToFloatArray(samples);
        internalTestCostFunctionNoNormalization(nn, false, singlePrecisionSamples);
    }

    @Test
    public void testSinglePrecisionGradient() throws IOException {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.SINGLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = listDoubleArrayToListFloatArray(weights);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        float[] singlePrecisionSamples = doubleArrayToFloatArray(samples);
        internalTestGradient(nn, false, singlePrecisionSamples);
    }

    @Test
    public void testDoublePrecisionGradient() throws IOException {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        nnd.weightsArrays = weights;
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestGradient(nn, true, samples);
    }
    
    private void internalTestGradient(NeuralNetwork nn, boolean useDoublePrecision, Object samples) throws IOException {
        boolean passed = true;

        if (useDoublePrecision)
            nn.run((double[])samples);
        else
            nn.run((float[])samples);
        nn.setLablesVector(labels);
        LinkedList<cl_mem> gradients = nn.calculateGradient(0.5f);

        NetworkDescription nnd = nn.getNetworkDescription();
        cl_mem clmWeightsGradMatrix2 = gradients.get(1);
        int weightsWidth2 = nnd.weightsWidths.get(nnd.weightsWidths.size() - 1);
        int size = weightsWidth2 * nnd.weightsHeights.get(nnd.weightsHeights.size() - 1);
        Object weightsGrad2;
        if (nnd.getPrecision().isDouble()) {
            weightsGrad2 = new double[size];
            clEnqueueReadBuffer(nn.computeDevice.getCommandQueue(), clmWeightsGradMatrix2, CL_TRUE, 0,
                    nn.computeDevice.bufferSize(size), Pointer.to((double[])weightsGrad2), 0, null, null);
        }
        else {
            weightsGrad2 = new float[size];
            clEnqueueReadBuffer(nn.computeDevice.getCommandQueue(), clmWeightsGradMatrix2, CL_TRUE, 0,
                    nn.computeDevice.bufferSize(size), Pointer.to((float[])weightsGrad2), 0, null, null);
        }
        clReleaseMemObject(clmWeightsGradMatrix2);

        BufferedReader weightsGradFile2 = new BufferedReader(new InputStreamReader(
                getClass().getClassLoader().getResourceAsStream("WeightsGradient2.txt")));
        String line = weightsGradFile2.readLine();
        int i = 0;
        while (line != null) {
            double v = nnd.getPrecision().isDouble() ? ((double[])weightsGrad2)[i] : ((float[])weightsGrad2)[i];
            if (Double.isNaN(v) || Math.abs(v - Double.parseDouble(line)) > 0.000001) {
                passed = false;
                break;
            }
            i++;
            line = weightsGradFile2.readLine();
        }
        weightsGradFile2.close();
        
        if (passed) {
            cl_mem clmWeightsGradMatrix1 = gradients.get(0);
            int weightsWidth1 = nnd.weightsWidths.get(nnd.weightsWidths.size() - 2);
            size = weightsWidth1 * nnd.weightsHeights.get(nnd.weightsHeights.size() - 2);
            Object weightsGrad1;
            if (nnd.getPrecision().isDouble()) {
                weightsGrad1 = new double[size];
                clEnqueueReadBuffer(nn.computeDevice.getCommandQueue(), clmWeightsGradMatrix1, CL_TRUE, 0,
                        nn.computeDevice.bufferSize(size), Pointer.to((double[])weightsGrad1), 0, null, null);
            }
            else {
                weightsGrad1 = new float[size];
                clEnqueueReadBuffer(nn.computeDevice.getCommandQueue(), clmWeightsGradMatrix1, CL_TRUE, 0,
                        nn.computeDevice.bufferSize(size), Pointer.to((float[])weightsGrad1), 0, null, null);
            }
            clReleaseMemObject(clmWeightsGradMatrix1);
            BufferedReader weightsGradFile1 = new BufferedReader(new InputStreamReader(
                    getClass().getClassLoader().getResourceAsStream("WeightsGradient1.txt")));
            line = weightsGradFile1.readLine();
            i = 0;
            while (line != null) {
                double v = nnd.getPrecision().isDouble() ? ((double[])weightsGrad1)[i] : ((float[])weightsGrad1)[i];
                if (Double.isNaN(v) || Math.abs(v - Double.parseDouble(line)) > 0.000001) {
                    passed = false;
                    break;
                }
                i++;
                line = weightsGradFile1.readLine();
            }
            weightsGradFile1.close();
        }
        
        assertTrue(passed);
        nn.release();
    }

    @Test
    public void testTrainingWithSinglePrecisionOnSinglePrecisionSample() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.SINGLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        float[] singlePrecisionSamples = doubleArrayToFloatArray(samples);
        internalTestTraining(nn, false, singlePrecisionSamples, labels);
    }

    @Test
    public void testTrainingWithSinglePrecisionOnDoublePrecisionSample() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.SINGLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestTraining(nn, true, samples, labels);
    }
    
    @Test
    public void testTrainingWithDoublePrecisionOnDoublePrecisionSample() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        internalTestTraining(nn, true, samples, labels);
    }
    
    @Test
    public void testTrainingWithDoublePrecisionOnSinglePrecisionSample() {
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, numOfFeatures, numOfLayers, numOfNodesInLayer, numOfLabelCategories);
        NeuralNetwork nn = new NeuralNetwork(platformID, deviceID, nnd);
        float[] singlePrecisionSamples = doubleArrayToFloatArray(samples);
        internalTestTraining(nn, false, singlePrecisionSamples, labels);
    }
    
    private void internalTestTraining(NeuralNetwork nn, boolean useDoublePrecision, Object samples, int[] labels) {
        int numOfIterations = 50;
        double lambda = 0.0f;
        
        TrainingProgress progress = new TrainingProgress() {
            public void reportProgress(int iteration, double cost) {
                System.out.println("Iteration " + iteration + ": Cost " + cost);
            }
        };
        Object resultMatrix;
        if (useDoublePrecision) {
            nn.train((double[])samples, labels, numOfIterations, lambda, progress);
            resultMatrix = nn.run((double[])samples);
        }
        else {
            nn.train((float[])samples, labels, numOfIterations, lambda, progress);
            resultMatrix = nn.run((float[])samples);
        }
        
        int numOfCorrectPrediction = 0;
        for (int i = 0; i < numOfSamples; i++) {
            int maxProbIndex = 0;
            for (int j = 1; j < numOfLabelCategories; j++) {
                double current = useDoublePrecision ?
                        ((double[])resultMatrix)[i * numOfLabelCategories + j] :
                        ((float[])resultMatrix)[i * numOfLabelCategories + j];     
                double max = useDoublePrecision ?
                        ((double[])resultMatrix)[i * numOfLabelCategories + maxProbIndex] :
                        ((float[])resultMatrix)[i * numOfLabelCategories + maxProbIndex];     
                if (current > max)
                    maxProbIndex = j;
            }
            int prediction = maxProbIndex;
            if (prediction == labels[i])
                numOfCorrectPrediction++;
        }
        double precision = (((double)numOfCorrectPrediction) / numOfSamples) * 100;
        System.out.println("Correct Prediction Percentage: " + precision);

        assertTrue(Math.abs(precision - 95.0) < 5.0);
        nn.release();
    }

    /**
     * Trains and runs a sample image recognition neural network
     */
    public static void main(String[] args) throws Exception {
        // Print out the OpecnCL environment
        System.out.println("\n#####################################################################\n");
        TestComputeDevice.main(args);
        
        // Load the test data
        TestNeuralNetwork instance = new TestNeuralNetwork();
        instance.loadData();
        
        System.out.println("\nRunning the image recognition neural network on OpenCL platform " + 
                instance.platformID + " device " + instance.deviceID + ".");
        System.out.println("Set custom OpenCL platform and device via the anna.platformID and " + 
                "anna.deviceID Java system properties.\n");
        
        // Train a neural network on the test data
        NetworkDescription nnd = new NetworkDescription(
                FloatingPointPrecision.DOUBLE, instance.numOfFeatures, instance.numOfLayers, 
                instance.numOfNodesInLayer, instance.numOfLabelCategories);
        NeuralNetwork nn = new NeuralNetwork(instance.platformID, instance.deviceID, nnd);
        TrainingProgress progress = new TrainingProgress() {
            public void reportProgress(int iteration, double cost) {
                System.out.print(".");
            }
        };
        System.out.println("Training begins");
        nn.train(instance.samples, instance.labels, 100, 0.0, progress);
        System.out.println("\nTraining completed!\n");
        
        // Pick a random sample image
        Random rnd = new Random();
        int randomSample = rnd.nextInt(instance.numOfSamples);
        int firstIndex = randomSample * instance.numOfFeatures;
        double[] bitmap = Arrays.copyOfRange(instance.samples, firstIndex, firstIndex + instance.numOfFeatures);

        // Print the random image in ASCII
        System.out.println("Recognizing the following image:");
        int x = 0; // x axis 
        int y = 0; // y axis
        for (int i = 1; i <= instance.numOfFeatures; i++) {
            double pixel = bitmap[y * 20 + x]; // test bitmap images were encoded with axes inverted 
            System.out.print(pixel > 0.5 ? '*' : ' '); // loose shades of grey
            if (i % 20 == 0) { // each image is 20x20 bitmap
                x++;
                y = 0;
                System.out.println();
            } 
            else {
                y++;
            }
        }
        
        // Run the neural network on the random image
        double[] vector = nn.run(bitmap);
        int prediction = 0;
        double max = vector[0];
        for (int i = 1; i < vector.length; i++) {
            if (vector[i] > max) {
                max = vector[i];
                prediction = i;
            }
        }

        System.out.println("\nShown digit: " + correctLabel(instance.labels[randomSample]));
        System.out.println("Predicted digit: " + correctLabel(prediction));
        System.out.println("\n#####################################################################\n");

        nn.release();
    }
    
    // Due to an indexing mismatch in the test data the label 9 must be turned into 0 and 
    // other labels must be incremented by 1   
    static int correctLabel(int digit) {
        return digit == 9 ? 0 : digit + 1;
    }

}
