package org.anna;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

/**
 * This class is used when creating a new {@link NeuralNetwork} instance. It describes the neural network with 
 * the following properties:
 * <p>
 * <ul>
 * <li>precision - floating point precision used for math calculations (set to SINGLE for speed or DOUBLE for accuracy)
 * <li>numOfFeatures - number of columns (or dimensions) in the input matrix (excluding the label column)
 * <li>numOfLayers - number of layers in the neural network (should be at least 2)
 * <li>numOfNodesInLayer - number of nodes in each layer (except the last one which always has numOfLabelCategories nodes)
 * <li>numOfLabelCategories - number of distinct label classes (use 2 for binary classification)
 * </ul>
 * <p>
 * Once the neural network is trained, an instance of this class is also used to store the neural network's weights.
 * The weights are internal and cannot be accessed explicitly. However, this class is serializable and the weights 
 * are saved/restored along with all other properties of this class.     
 * 
 * @see NeuralNetwork#getNetworkDescription()
 * @author Eugene Ivanov
 *
 */
public class NetworkDescription implements Serializable {
    private static final long serialVersionUID = 1L;
    
    FloatingPointPrecision precision;
    int numOfFeatures;
    int numOfLayers;
    int numOfNodesInLayer;
    int numOfLabelCategories;
    
    /**
     * Use this constructor to create a new neural network description. See the class description on the parameters details.
     */
    @SuppressWarnings("unchecked")
    public NetworkDescription(FloatingPointPrecision precision, int numOfFeatures, int numOfLayers, 
            int numOfNodesInLayer, int numOfLabelCategories) {
        if (precision == null)
            throw new IllegalArgumentException("precision should not be null");
        if (numOfFeatures <= 0) 
            throw new IllegalArgumentException("numOfFeatures should be greater than 0");
        if (numOfLayers <= 1) 
            throw new IllegalArgumentException("numOfLayers should be at least 2");
        if (numOfNodesInLayer <= 0) 
            throw new IllegalArgumentException("numOfNodesInLayer should be greater than 0");
        if (numOfLabelCategories <= 1) 
            throw new IllegalArgumentException("numOfLabelCategories should be at least 2");

        this.precision = precision;
        this.numOfFeatures = numOfFeatures;
        this.numOfLayers = numOfLayers;
        this.numOfNodesInLayer = numOfNodesInLayer;
        this.numOfLabelCategories = numOfLabelCategories;
        
        // Create weights matrices 
        for (int k = 0; k < numOfLayers; k++) {
            int height = 0;
            int width = 0;
            if (k == 0) {
                height = numOfFeatures + 1; // plus one row for bias
                width = numOfNodesInLayer;
            }
            else if (k == numOfLayers - 1) {
                height = numOfNodesInLayer + 1; // plus one row for bias
                width = numOfLabelCategories;
            }
            else {
                height = numOfNodesInLayer + 1; // plus one row for bias
                width = numOfNodesInLayer;
            }
            weightsHeights.add(height);
            weightsWidths.add(width);

            // Initialize weights with random numbers per a recommendation from Andrew Ng's Machine Learning class
            Object array;
            array = precision.isDouble() ? new double[height * width] : new float[height * width];
            double epsilon = 2.45f / Math.sqrt(height - 1 + width); // 2.45 ~ sqrt(6)
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    if (precision.isDouble())
                        ((double[])array)[i * width + j] = (rnd.nextDouble() * epsilon * 2) - epsilon;
                    else
                        ((float[])array)[i * width + j] = (float)((rnd.nextDouble() * epsilon * 2) - epsilon);
                }
            }
            weightsArrays.add(array);
        }
    }

    public FloatingPointPrecision getPrecision() {
        return precision;
    }
    
    public int getNumOfFeatures() {
        return numOfFeatures;
    }

    public int getNumOfLayers() {
        return numOfLayers;
    }

    public int getNumOfNodesInLayer() {
        return numOfNodesInLayer;
    }

    public int getNumOfLabelCategories() {
        return numOfLabelCategories;
    }

    // List of arrays of real numbers (double[] or float[] depending on precision) defining weight matrices 
    @SuppressWarnings("rawtypes")
    ArrayList weightsArrays = new ArrayList();

    ArrayList<Integer> weightsHeights = new ArrayList<Integer>(); 
    ArrayList<Integer> weightsWidths = new ArrayList<Integer>();

    // Used to assign initial weights
    private static Random rnd = new Random(System.currentTimeMillis());
}
