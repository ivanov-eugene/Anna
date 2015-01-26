package org.anna;

/**
 * An implementation of this interface receives notifications about neural network training progress. 
 * 
 * @author Eugene Ivanov
 *
 */
public interface TrainingProgress {
    /**
     * This method is invoked every time the training algorithm moves on to a new iteration.
     * @param iteration current iteration number
     * @param cost the cost associated with the current iteration (the cost gets lower with the training progress)   
     */
    void reportProgress(int iteration, double cost);
}