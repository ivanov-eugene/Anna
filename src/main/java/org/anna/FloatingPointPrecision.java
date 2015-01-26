package org.anna;

/**
 * Floating point precision used for math calculations (use SINGLE for speed or DOUBLE for accuracy).
 * 
 * @author Eugene Ivanov
 *
 */
public enum FloatingPointPrecision {
    SINGLE,
    DOUBLE;
    
    public boolean isDouble() { return this == DOUBLE; }
}
