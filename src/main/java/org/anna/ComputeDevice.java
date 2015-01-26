package org.anna;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jocl.*;

import static org.jocl.CL.*;

/**
 * A utility class providing access to OpenCL kernels helping to speed up Neural Network's forward and backward
 * propagation algorithms. The class utilizes a mixture of the singleton and factory design patterns:
 * to get an instance of the class, use the static getDeviceInstance() function. 
 *   
 * @author Eugene Ivanov
 */
public class ComputeDevice {
    /**
     * Returns a singleton ComputeDevice instance for a particular combination of OpenCL platform and device.
     * 
     * @param platformIndex OpenCL platform index starting with 0
     * @param deviceIndex OpenCL device index starting with 0
     * @param precision use double precision type (double) if true or single precision type (float) if false
     */
    public static ComputeDevice getDeviceInstance(int platformIndex, int deviceIndex, FloatingPointPrecision precision) {
        // Make sure that the activation happens only once and it is thread safe 
        synchronized (activationLock) {

            // Enable throwing org.ocl.CLException runtime exceptions
            CL.setExceptionsEnabled(true);

            // Obtain the number of platforms
            int numPlatformsArray[] = new int[1];
            clGetPlatformIDs(0, null, numPlatformsArray);
            int numPlatforms = numPlatformsArray[0];
            if (platformIndex < 0 || platformIndex >= numPlatforms)
                throw new IllegalArgumentException("Illegal platform index value");

            // Obtain a platform ID
            cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
            clGetPlatformIDs(platforms.length, platforms, null);
            cl_platform_id platform = platforms[platformIndex];

            // Obtain the number of devices for the platform
            int numDevicesArray[] = new int[1];
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
            int numDevices = numDevicesArray[0];
            if (deviceIndex < 0 || deviceIndex >= numDevices)
                throw new IllegalArgumentException("Illegal device index value");

            // Obtain a device ID 
            cl_device_id devices[] = new cl_device_id[numDevices];
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, null);
            cl_device_id device = devices[deviceIndex];

            // Return ComputeDevice if it already exists and active
            if (deviceMap.get(platformIndex) == null) {
                deviceMap.put(platformIndex, new HashMap<Integer,ComputeDevice>());
            }
            ComputeDevice instance = deviceMap.get(platformIndex).get(deviceIndex);
            if (instance != null && instance.active) {
                return instance;
            }

            // Create and initialize a new ComputeDevice instance
            instance = new ComputeDevice(platform, device, precision);
            deviceMap.get(platformIndex).put(deviceIndex, instance);                
            return instance;
        }
    }

    // Private constructor to enforce Factory pattern
    private ComputeDevice(cl_platform_id platform, cl_device_id device, FloatingPointPrecision precision) {
        this.precision = precision;

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, new cl_device_id[] {device}, null, null, null);

        // Create a command-queue for the selected device
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        // Create a special empty buffer used to indicate NULL pointer
        clmNullDummy = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_int, null, null);
        clEnqueueReadBuffer(commandQueue, clmNullDummy, CL_TRUE, 0,
                Sizeof.cl_int, Pointer.to(new double[] {0}), 0, null, null);

        // Read, process and build the NeuralNetwork.cl program 
        HashMap<String, String> substitutions = new HashMap<String, String>();
        substitutions.put("(#define REAL )(\\w+)", precision.isDouble() ? "double" : "float"); 
        String source = readAndProcessSourceFile("NeuralNetwork.cl", substitutions);
        program = clCreateProgramWithSource(context, 1, new String[] {source}, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        // Create kernel functions from the program
        kernelMatrixMultiplication = clCreateKernel(program, "matrixMultiplication", null);
        kernelVectorTotal = clCreateKernel(program, "vectorTotalByReduce", null);
        kernelVectorSquaredTotal = clCreateKernel(program, "vectorSquaredTotalByReduce", null);
        kernelForwardPropagation = clCreateKernel(program, "forwardPropagation", null);
        krenelCostFunctionPartLog = clCreateKernel(program, "costFunctionPartLog", null);
        kernelCalculateGradientTheta = clCreateKernel(program, "calculateGradientTheta", null);
        kernelCalculateGradientDelta = clCreateKernel(program, "calculateGradientDelta", null);

        active = true;
    }

    // If false, single precision is used
    private FloatingPointPrecision precision;

    /**
     * Returns OpenCL context used by this object.
     */
    public cl_context getContext() {
        return context;
    }

    // OpenCL context used by this object
    private cl_context context = null;

    /**
     * Returns OpenCL Command Queue used by this object.
     */
    public cl_command_queue getCommandQueue() {
        return commandQueue;
    }

    // OpenCL Command Queue used by this object
    private cl_command_queue commandQueue = null;

    // OpenCL Program used by this object
    private cl_program program = null;

    // OpenCL kernels used by this object
    private cl_kernel kernelMatrixMultiplication = null;
    private cl_kernel kernelForwardPropagation = null;
    private cl_kernel kernelVectorTotal = null;
    private cl_kernel kernelVectorSquaredTotal = null;
    private cl_kernel krenelCostFunctionPartLog = null;
    private cl_kernel kernelCalculateGradientTheta = null;
    private cl_kernel kernelCalculateGradientDelta = null;

    // Used to pass null values as OpenCL does not allow null explicitly
    private cl_mem clmNullDummy = null; 

    // A synchronization lock used to implement a thread safe factory 
    private static Object activationLock = new Object();

    // A flag indicating whether the object is active or released
    private boolean active = false;

    // Map: platform index -> device index -> ComputeDevice instance
    private static HashMap<Integer, HashMap<Integer, ComputeDevice>> deviceMap = 
            new HashMap<Integer, HashMap<Integer,ComputeDevice>>();

    /**
     * Release all OpenCL artifacts used by this object. 
     */
    public synchronized void release() {
        if (active) {
            if (clmNullDummy != null)
                clReleaseMemObject(clmNullDummy);
            if (kernelMatrixMultiplication != null)
                clReleaseKernel(kernelMatrixMultiplication);
            if (kernelVectorTotal != null)
                clReleaseKernel(kernelVectorTotal);
            if (kernelVectorSquaredTotal != null)
                clReleaseKernel(kernelVectorSquaredTotal);
            if (kernelForwardPropagation != null)
                clReleaseKernel(kernelForwardPropagation);
            if (krenelCostFunctionPartLog != null)
                clReleaseKernel(krenelCostFunctionPartLog);
            if (kernelCalculateGradientTheta != null)
                clReleaseKernel(kernelCalculateGradientTheta);
            if (kernelCalculateGradientDelta != null)
                clReleaseKernel(kernelCalculateGradientDelta);
            if (program != null)
                clReleaseProgram(program);
            if (commandQueue != null)
                clReleaseCommandQueue(commandQueue);
            if (context != null)
                clReleaseContext(context);
            active = false;
        }
    }
    
    private void confirmActiveState() {
        if (!active) {
            throw new IllegalStateException("The object was released and can no longer be used");
        }
    }

    // Read an OpenCL source file packaged as a resource and do some text substitutions
    private String readAndProcessSourceFile(String fileName, HashMap<String, String> substitutions) {
        try {
            // Read the source file packaged as a resource
            BufferedReader br = new BufferedReader(new InputStreamReader(
                    getClass().getClassLoader().getResourceAsStream(fileName)));
            StringBuffer sb = new StringBuffer();
            try {
                String line = null;
                while (true) {
                    line = br.readLine();
                    if (line == null) {
                        break;
                    }
                    sb.append(line + "\n");
                }
            }
            finally {
                br.close();
            }
            String text = sb.toString();

            // Handle substitutions as key-value pairs where key is a regular expression and value is the substitution
            if (substitutions != null) {
                for (Entry<String, String> s : substitutions.entrySet()) {
                    Pattern p = Pattern.compile(s.getKey());
                    Matcher m = p.matcher(text);
                    sb = new StringBuffer();
                    while (m.find()) {
                        m.appendReplacement(sb, m.group(1) + s.getValue());
                    }
                    m.appendTail(sb);
                    text = sb.toString();
                }
            }

            return text;
        }
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Calculate buffer size in bytes depending of the precision (8 bytes for double, 4 bytes for float).  
     * @param arrayLength buffer size in elements
     * @return buffer size in bytes
     */
    public int bufferSize(int arrayLength) {
        return arrayLength * (precision.isDouble() ? Sizeof.cl_double : Sizeof.cl_float); 
    }

    /**
     * Invoke matrix multiplication kernel. 
     * @param matrixA input matrix A as OpenCL buffer
     * @param matrixB input matrix B as OpenCL buffer
     * @param matrixC result matrix C as OpenCL buffer
     * @param widthA width of matrix A 
     * @param heightA height of matrix A 
     * @param widthB width of matrix B
     */
    public synchronized void matrixMultiplication(cl_mem matrixA, cl_mem matrixB, cl_mem matrixC, 
            int widthA, int heightA, int widthB) {
        confirmActiveState();
        
        clSetKernelArg(kernelMatrixMultiplication, 1, Sizeof.cl_mem, Pointer.to(matrixB));
        clSetKernelArg(kernelMatrixMultiplication, 0, Sizeof.cl_mem, Pointer.to(matrixA));
        clSetKernelArg(kernelMatrixMultiplication, 2, Sizeof.cl_mem, Pointer.to(matrixC));
        clSetKernelArg(kernelMatrixMultiplication, 3, Sizeof.cl_int, Pointer.to(new int[] {widthA}));
        clSetKernelArg(kernelMatrixMultiplication, 4, Sizeof.cl_int, Pointer.to(new int[] {widthB}));

        long globalThreads[] = new long[] {widthB, heightA};
        clEnqueueNDRangeKernel(commandQueue, kernelMatrixMultiplication, 
                2, null, globalThreads, null, 0, null, null);
    }

    /**
     * Invoke the kernel implementing one step of the forward propagation algorithm.
     * @param inMatrix 
     * @param thetaMatrix
     * @param outMatrix
     * @param firstThetaBiasIndex
     * @param inWidth
     * @param inHeight
     * @param thetaWidth
     */
    public synchronized void forwardPropagation(cl_mem inMatrix, cl_mem thetaMatrix, cl_mem outMatrix, 
            int firstThetaBiasIndex, int inWidth, int inHeight, int thetaWidth) {
        confirmActiveState();
        
        clSetKernelArg(kernelForwardPropagation, 0, Sizeof.cl_mem, Pointer.to(inMatrix));
        clSetKernelArg(kernelForwardPropagation, 1, Sizeof.cl_mem, Pointer.to(thetaMatrix));
        clSetKernelArg(kernelForwardPropagation, 2, Sizeof.cl_mem, Pointer.to(outMatrix));
        clSetKernelArg(kernelForwardPropagation, 3, Sizeof.cl_int, Pointer.to(new int[] {firstThetaBiasIndex}));
        clSetKernelArg(kernelForwardPropagation, 4, Sizeof.cl_int, Pointer.to(new int[] {inWidth}));

        long globalThreads[] = new long[] {thetaWidth, inHeight};
        clEnqueueNDRangeKernel(commandQueue, kernelForwardPropagation, 
                2, null, globalThreads, null, 0, null, null);
    }

    /**
     * Invoke the kernel calculating a total of all vector elements. 
     * @param clmVector input vector as OpenCL buffer
     * @param vectorLength number of elements in the input vector
     * @return total as double
     */
    public synchronized double calculateTotal(cl_mem clmVector, int vectorLength) {
        return internalCalculateTotal(kernelVectorTotal, clmVector, vectorLength);
    }

    /**
     * Invoke the kernel calculating a total of all vector elements' squares. 
     * @param clmVector input vector as OpenCL buffer
     * @param vectorLength number of elements in the input vector
     * @return total as double
     */
    public synchronized double calculateSquaredTotal(cl_mem clmVector, int vectorLength) {
        return internalCalculateTotal(kernelVectorSquaredTotal, clmVector, vectorLength);
    }

    private double internalCalculateTotal(cl_kernel kernel, cl_mem clmVector, int vectorLength) {
        confirmActiveState();
        if (vectorLength < 1) throw new IllegalArgumentException("vectorLength should be larger than 0");

        int localGroupSize = 64; // TODO make it configurable
        long localThreads[] = new long[] {localGroupSize};
        int numOfPairs = (vectorLength + 1) / 2;
        int numOfWorkgroups = (numOfPairs + localGroupSize - 1) / localGroupSize;
        int numOfGlobalThreads = numOfWorkgroups * localGroupSize;
        long globalThreads[] = new long[] {numOfGlobalThreads};
        int reminder = numOfPairs % localGroupSize;
        int firstZeroIndex = (reminder == 0 ? localGroupSize : reminder) * 2;
        if (vectorLength % 2 == 1)
            firstZeroIndex--;

        cl_mem clmOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, bufferSize(numOfWorkgroups), null, null);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(clmVector));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(clmOutput));
        clSetKernelArg(kernel, 2, Sizeof.cl_double * localGroupSize, null);
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] {firstZeroIndex}));

        clEnqueueNDRangeKernel(commandQueue, kernel, 
                1, null, globalThreads, localThreads, 0, null, null);

        // Read the output data
        double sum = 0.0;
        if (precision.isDouble()) {
            double[] output = new double[numOfWorkgroups];
            clEnqueueReadBuffer(commandQueue, clmOutput, CL_TRUE, 0,
                    bufferSize(numOfWorkgroups), Pointer.to(output), 0, null, null);
            for (int i = 0; i < output.length; i++) {
                sum += output[i];
            }
        }
        else {
            float[] output = new float[numOfWorkgroups];
            clEnqueueReadBuffer(commandQueue, clmOutput, CL_TRUE, 0,
                    bufferSize(numOfWorkgroups), Pointer.to(output), 0, null, null);
            for (int i = 0; i < output.length; i++) {
                sum += output[i];
            }
        }
        clReleaseMemObject(clmOutput);
        return sum;
    }

    /**
     * Invoke the kernel implementing a part of the cost function calculations  
     * @param clmVectorY
     * @param clmMatrixA
     * @param clmMatrixLogA
     * @param height
     * @param width
     */
    public synchronized void calculateLogPartCostFunction(cl_mem clmVectorY, cl_mem clmMatrixA, cl_mem clmMatrixLogA,
            int height, int width) {
        confirmActiveState();
        
        clSetKernelArg(krenelCostFunctionPartLog, 0, Sizeof.cl_mem, Pointer.to(clmVectorY));
        clSetKernelArg(krenelCostFunctionPartLog, 1, Sizeof.cl_mem, Pointer.to(clmMatrixA));
        clSetKernelArg(krenelCostFunctionPartLog, 2, Sizeof.cl_mem, Pointer.to(clmMatrixLogA));

        long globalThreads[] = new long[] {width, height};
        clEnqueueNDRangeKernel(commandQueue, krenelCostFunctionPartLog, 
                2, null, globalThreads, null, 0, null, null);
    }

    /**
     * Invoke the kernel implementing a part of the back propagation algorithm.
     * @param clmMatrixA
     * @param clmMatrixPriorA
     * @param clmVectorY
     * @param clmMatrixTheta
     * @param clmMatrixResult
     * @param height
     * @param width
     * @param numOfSamples
     * @param lambda
     */
    public synchronized void calculateGradient(cl_mem clmMatrixA, cl_mem clmMatrixPriorA, cl_mem clmVectorY, 
            cl_mem clmMatrixTheta, cl_mem clmMatrixResult, int height, int width, int numOfSamples, double lambda) 
    {
        confirmActiveState();
        
        clSetKernelArg(kernelCalculateGradientTheta, 0, Sizeof.cl_mem, Pointer.to(clmMatrixA));
        clSetKernelArg(kernelCalculateGradientTheta, 1, Sizeof.cl_mem, Pointer.to(clmMatrixPriorA));
        clSetKernelArg(kernelCalculateGradientTheta, 2, Sizeof.cl_mem, 
                clmVectorY != null ? Pointer.to(clmVectorY) : Pointer.to(clmNullDummy));
        clSetKernelArg(kernelCalculateGradientTheta, 3, Sizeof.cl_mem, Pointer.to(clmMatrixTheta));
        clSetKernelArg(kernelCalculateGradientTheta, 4, Sizeof.cl_mem, Pointer.to(clmMatrixResult));
        clSetKernelArg(kernelCalculateGradientTheta, 5, Sizeof.cl_int, Pointer.to(new int[] {numOfSamples}));
        if (precision.isDouble())
            clSetKernelArg(kernelCalculateGradientTheta, 6, Sizeof.cl_double, Pointer.to(new double[] {lambda}));
        else
            clSetKernelArg(kernelCalculateGradientTheta, 6, Sizeof.cl_float, Pointer.to(new float[] {(float)lambda}));
        clSetKernelArg(kernelCalculateGradientTheta, 7, Sizeof.cl_int, 
                Pointer.to(new int[] {clmVectorY != null ? 1 : 0}));

        long globalThreads[] = new long[] {height, width};
        clEnqueueNDRangeKernel(commandQueue, kernelCalculateGradientTheta, 
                2, null, globalThreads, null, 0, null, null);
    }

    /**
     * Invoke the kernel implementing a part of the back propagation algorithm.
     * @param clmMatrixPriorDelta
     * @param clmVectorY
     * @param clmMatrixTheta
     * @param clmMatrixA
     * @param clmMatrixResultDelta
     * @param priorDeltaWidth
     * @param resultDeltaWidth
     * @param numOfSamples
     */
    public synchronized void calculateGradientDelta(cl_mem clmMatrixPriorDelta, cl_mem clmVectorY, 
            cl_mem clmMatrixTheta, cl_mem clmMatrixA, cl_mem clmMatrixResultDelta, 
            int priorDeltaWidth, int resultDeltaWidth, int numOfSamples) {
        confirmActiveState();
        
        clSetKernelArg(kernelCalculateGradientDelta, 0, Sizeof.cl_mem, Pointer.to(clmMatrixPriorDelta));
        clSetKernelArg(kernelCalculateGradientDelta, 1, Sizeof.cl_mem, 
                clmVectorY != null ? Pointer.to(clmVectorY) : Pointer.to(clmNullDummy));
        clSetKernelArg(kernelCalculateGradientDelta, 2, Sizeof.cl_mem, Pointer.to(clmMatrixTheta));
        clSetKernelArg(kernelCalculateGradientDelta, 3, Sizeof.cl_mem, Pointer.to(clmMatrixA));
        clSetKernelArg(kernelCalculateGradientDelta, 4, Sizeof.cl_mem, Pointer.to(clmMatrixResultDelta));
        clSetKernelArg(kernelCalculateGradientDelta, 5, Sizeof.cl_int, Pointer.to(new int[] {priorDeltaWidth}));
        clSetKernelArg(kernelCalculateGradientDelta, 6, Sizeof.cl_int, 
                Pointer.to(new int[] {clmVectorY != null ? 1 : 0}));

        long globalThreads[] = new long[] {numOfSamples, resultDeltaWidth};
        clEnqueueNDRangeKernel(commandQueue, kernelCalculateGradientDelta, 
                2, null, globalThreads, null, 0, null, null);
    }

}
