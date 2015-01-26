package org.anna;

import java.util.Random;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_device_id;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.junit.Test;

import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_PLATFORM_NAME;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clGetPlatformInfo;
import static org.jocl.CL.clReleaseMemObject;
import static org.junit.Assert.*;

public class TestComputeDevice {

    @Test
    public void testSameInstance() {
        ComputeDevice d1 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);
        ComputeDevice d2 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);
        assertTrue(d1 == d2);
        d1.release();
        d2.release();
    }

    @Test
    public void testDifferentInstance() {
        ComputeDevice d1 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);
        ComputeDevice d2 = ComputeDevice.getDeviceInstance(0, 1, FloatingPointPrecision.DOUBLE);
        assertTrue(d1 != d2);
        d1.release();
        d2.release();
    }

    @Test
    public void testDeactivatedInstance() {
        ComputeDevice d1 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);
        d1.release();
        ComputeDevice d2 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);
        assertTrue(d1 != d2);
        d2.release();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongPlatform() {
        // there should be just a few platforms
        ComputeDevice.getDeviceInstance(9999, 0, FloatingPointPrecision.DOUBLE);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testWrongDevice() {
        // there should be just a few devices
        ComputeDevice.getDeviceInstance(0, 9999, FloatingPointPrecision.DOUBLE);
    }

    @Test
    public void testDoublePrecision() {
        ComputeDevice d1 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);
        assertTrue(d1.bufferSize(1) == Sizeof.cl_double);
        d1.release();
    }

    @Test
    public void testSinglePrecision() {
        ComputeDevice d1 = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.SINGLE);
        assertTrue(d1.bufferSize(1) == Sizeof.cl_float);
        d1.release();
    }

    private Random rnd = new Random(System.currentTimeMillis());

    private void initArrayWithRandomNumbers(double[] array) {
        for(int i = 0; i < array.length; i++) {
            array[i] = rnd.nextDouble();
        }
    }

    @Test
    public void testMatrixMultiplication()  {
        ComputeDevice device = ComputeDevice.getDeviceInstance(0, 0, FloatingPointPrecision.DOUBLE);

        int heightA = 10;
        int widthA = 11;
        int widthB = 12;
        int heightB = widthA;
        int widthC = widthB;
        int heightC = heightA;

        double[] matrixA = new double[widthA * heightA];
        double[] matrixB = new double[widthB * heightB];
        double[] matrixC = new double[widthC * heightC];

        initArrayWithRandomNumbers(matrixA);
        initArrayWithRandomNumbers(matrixB);

        cl_mem clmMatrixA = clCreateBuffer(device.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                device.bufferSize(matrixA.length), Pointer.to(matrixA), null);
        cl_mem clmMatrixB = clCreateBuffer(device.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                device.bufferSize(matrixB.length), Pointer.to(matrixB), null);
        cl_mem clmMatrixC = clCreateBuffer(device.getContext(), CL_MEM_READ_WRITE, 
                device.bufferSize(matrixC.length), null, null);

        device.matrixMultiplication(clmMatrixA, clmMatrixB, clmMatrixC, widthA, heightA, widthB);

        // Read the output data
        clEnqueueReadBuffer(device.getCommandQueue(), clmMatrixC, CL_TRUE, 0,
                device.bufferSize(matrixC.length), Pointer.to(matrixC), 0, null, null);

        // Verify result
        boolean passed = true;
        for (int i = 0; i < heightC; i++) {
            for (int j = 0; j < widthC; j++) {
                double s = 0;
                for (int k = 0; k < widthA; k++) {
                    s += matrixA[i * widthA + k] * matrixB[k * widthB + j];
                }
                if (Math.abs(matrixC[i * widthC + j] - s) > 0.0001) {
                    passed = false;
                    break;
                }
            }
        }

        // Release kernel, program, and memory objects
        clReleaseMemObject(clmMatrixA);
        clReleaseMemObject(clmMatrixB);
        clReleaseMemObject(clmMatrixC);
        device.release();

        assertTrue(passed);
    }

    /**
     * Print out all OpenCL platforms and devices available on this machine.  
     */
    public static void main(String[] args) {
        // Obtain the number of platforms
        int numOfPlatforms[] = new int[1];
        clGetPlatformIDs(0, null, numOfPlatforms);

        if (numOfPlatforms[0] == 0) {
            System.out.println("No OpenCL platforms were found on this machine. " +
                    "Make sure that at least one OpenCL driver is installed and/or " +
                    "a compatible version of JVM (32-bit vs. 64-bit) is used " +
                    "(e.g. Intel 32-bit OpenCL driver is not visible to a 64-bit mode JVM).");
            System.exit(1);
            return;
        }

        System.out.println("This machine has the following OpenCL environment:");

        // Obtain the platform IDs
        cl_platform_id platforms[] = new cl_platform_id[numOfPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        // Collect all devices of all platforms
        for (int i = 0; i < platforms.length; i++) {
            // Obtain platform name
            long size[] = new long[1];
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, null, size);
            byte buffer[] = new byte[(int)size[0]];
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, buffer.length, Pointer.to(buffer), null);
            String platformName = new String(buffer, 0, buffer.length-1);
            System.out.println("Platform " + i + ": " + platformName);

            // Obtain all devices for the current platform
            int numOfDevices[] = new int[1];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, null, numOfDevices);
            cl_device_id devices[] = new cl_device_id[numOfDevices[0]];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numOfDevices[0], devices, null);
            for (int j = 0; j < devices.length; j++) {
                cl_device_id device = devices[j];
                clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size);
                buffer = new byte[(int)size[0]];
                clGetDeviceInfo(device, CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);
                String deviceName =  new String(buffer, 0, buffer.length-1);
                System.out.println("    Device " + j + ": " + deviceName);
            }
        }
    }
}
