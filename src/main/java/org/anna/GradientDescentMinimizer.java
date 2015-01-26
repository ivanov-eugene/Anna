package org.anna;

import java.util.Arrays;

/**
 * A generic gradient descent minimization utility based on an algorithm by 
 * Carl Edward Rasmussen. See the minimize() method for the copyright notice.
 *   
 * @author Eugene Ivanov
 *
 */
public abstract class GradientDescentMinimizer {

    public abstract double[] getWeights();
    public abstract void setWeights(double[] array);
    public abstract double[] advanceGradientDescent();
    public abstract double costFunction();
    public abstract void reportProgress(int iteration, double cost);

    double[] svmult(double s, double[] v) {
        double[] res = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            res[i] = v[i]*s;
        }
        return res;
    }

    double[] vvadd(double[] v1, double[] v2) {
        double[] res = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            res[i] = v1[i] + v2[i];
        }
        return res;
    }

    double scalar(double[] v1, double[] v2) {
        double res = 0;
        for (int i = 0; i < v1.length; i++) {
            res += v1[i] * v2[i];
        }
        return res;
    }

    /** 
     * Minimize a continuous differentialble multivariate function. Starting point
     * is given by "X" (D by 1), and the function named in the string "f", must
     * return a function value and a vector of partial derivatives. The Polack-
     * Ribiere flavour of conjugate gradients is used to compute search directions,
     * and a line search using quadratic and cubic polynomial approximations and the
     * Wolfe-Powell stopping criteria is used together with the slope ratio method
     * for guessing initial step sizes. Additionally a bunch of checks are made to
     * make sure that exploration is taking place and that extrapolation will not
     * be unboundedly large. The "length" gives the length of the run: if it is
     * positive, it gives the maximum number of line searches, if negative its
     * absolute gives the maximum allowed number of function evaluations. You can
     * (optionally) give "length" a second component, which will indicate the
     * reduction in function value to be expected in the first line-search (defaults
     * to 1.0). The function returns when either its length is up, or if no further
     * progress can be made (ie, we are at a minimum, or so close that due to
     * numerical problems, we cannot get any closer). If the function terminates
     * within a few iterations, it could be an indication that the function value
     * and derivatives are not consistent (ie, there may be a bug in the
     * implementation of your "f" function). The function returns the found
     * solution "X", a vector of function values "fX" indicating the progress made
     * and "i" the number of iterations (line searches or function evaluations,
     * depending on the sign of "length") used.
     * 
     * Copyright (C) 2001 and 2002 by Carl Edward Rasmussen.
     */
    public void minimize(int length) {
        double[] X = getWeights();

        double RHO=0.01f;                 //           % a bunch of constants for line searches
        double SIG=0.5f;     //  % RHO and SIG are the constants in the Wolfe-Powell conditions
        double INT=0.1f;    //% don't reevaluate within 0.1 of the limit of the current bracket
        double EXT=3.0f;      //              % extrapolate maximum 3 times the current bracket/*
        int MAXEVAL=20;         //                % max 20 function evaluations per line search
        double RATIO=100.0f;        //                              % maximum allowed slope ratio
        double MINDOUBLE = Double.MIN_VALUE;

        double[] X0;
        double f0;
        double[] df0;
        double red = 1;
        int iterations_i = 0;  
        int M;
        boolean ls_failed = false;
        double f1,f2; // cost
        double[] df2; // gradient
        double d2, f3, d3, z3, A, B, limit, z2;
        boolean success;

        double[] df1 = advanceGradientDescent();
        f1 = costFunction();
        reportProgress(iterations_i, f1);
        iterations_i = iterations_i + (length<0 ? 1 : 0);                 //count epochs?!  
        double[] s = svmult(-1.0f, df1); //search direction is steepest   
        double d1 = 0;
        for (int i = 0; i < X.length; i++) 
            d1 += -s[i] * s[i]; //this is the slope
        double z1 = red / (1.0f - d1); // initial step is red/(|s|+1)
        while (iterations_i < Math.abs(length)) {                             
            iterations_i = iterations_i + (length>0 ? 1 : 0);  //count iterations?!
            X0 = Arrays.copyOf(X, X.length);
            f0 = f1;
            df0 = Arrays.copyOf(df1, df1.length);//make a copy of current values
            X = vvadd(X, svmult(z1, s));        // begin line search
            setWeights(X);

            df2 = advanceGradientDescent();
            f2 = costFunction();
            reportProgress(iterations_i, f2);
            iterations_i = iterations_i + (length<0? 1 : 0);                 //count epochs?!          

            d2 = scalar(df2, s); 
            f3 = f1; 
            d3 = d1; 
            z3 = -z1;             // initialize point 3 equal to point 1                
            M = (length > 0)? MAXEVAL : (Math.min(MAXEVAL, -length - iterations_i));         

            success = false;
            limit = -1.0f;                    // initialize quanteties
            z2 = 0;
            while (true) {
                while (((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0)) {
                    limit = z1;
                    if (f2 > f1) {
                        z2 = z3 - (0.5f*d3*z3*z3)/(d3*z3+f2-f3);                 // quadratic fit
                    } 
                    else {
                        A = 6.0f*(f2-f3)/z3+3.0f*(d2+d3);                                // cubic fit
                        B = 3.0f*(f3-f2)-z3*(d3+2.0f*d2);
                        z2 = (double)(Math.sqrt(B*B-A*d2*z3*z3)-B)/A;      // numerical error possible - ok!
                    }
                    if (Double.isInfinite(z2) || Double.isNaN(z2)) 
                        z2 = z3/2.0f;
                    z2 = Math.max(Math.min(z2, INT*z3),(1.0f-INT)*z3);  // don't accept too close to limits
                    z1 = z1 + z2;                                           // update the step
                    X = vvadd(X, svmult(z2,s));
                    setWeights(X);

                    df2 = advanceGradientDescent();
                    f2 = costFunction();
                    reportProgress(iterations_i, f2);
                    M = M -1;
                    iterations_i = iterations_i + (length<0 ? 1 : 0);                 //count epochs?!

                    d2 = scalar(df2, s);
                    z3 = z3-z2;          
                } // end of while

                if ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) {
                    break;                           // this is a failure
                } else if (d2 > SIG*d1) {
                    success = true;
                    break;                                             // success
                }
                else if (M == 0) {
                    break;
                }
                A = 6.0f*(f2-f3)/z3+3.0f*(d2+d3);                      // make cubic extrapolation
                B = 3.0f*(f3-f2)-z3*(d3+2*d2);
                z2 = -d2*z3*z3/(B+(double)Math.sqrt(B*B-A*d2*z3*z3));        // num. error possible - ok!
                if  (Double.isNaN(z2) || Double.isInfinite(z2) || (z2 < 0)) {   // num prob or wrong sign?
                    if (limit < -0.5) {                             // if we have no upper limit
                        z2 = z1 * (EXT-1);                           // the extrapolate the maximum amount
                    } else {
                        z2 = (limit-z1)/2;                           // otherwise bisect
                    }
                } 
                else if ((limit > -0.5) & (z2+z1 > limit)) {     // extraplation beyond max?
                    z2 = (limit-z1)/2;    // bisect
                } else if ((limit < -0.5) & (z2+z1 > z1*EXT)) {       // extrapolation beyond limit
                    z2 = z1*(EXT-1.0f);                           // set to extrapolation limit
                } else if (z2 < -z3*INT) {
                    z2 = -z3*INT;
                } else if ((limit > -0.5) & (z2 < (limit-z1)*(1.0-INT)) )  { // too close to limit?
                    z2 = (limit-z1)*(1.0f-INT);
                }

                f3 = f2;
                d3 = d2;
                z3 = -z2;                  // set point 3 equal to point 2
                z1 = z1 + z2;
                X = vvadd(X, svmult(z2,s));                     // update current estimates   
                setWeights(X);
                df2 = advanceGradientDescent();
                f2 = costFunction();
                reportProgress(iterations_i, f2);
                M = M -1; 
                iterations_i = iterations_i + (length<0 ? 1 : 0);                 //count epochs?!

                d2 = scalar(df2, s);                        
            }     

            if (success) {                                        // if line search succeeded                   
                f1 = f2;
                // Polack-Ribiere direction
                s = vvadd(svmult((scalar(df2,df2) - scalar(df1,df2)) / (scalar(df1,df1)),s), svmult(-1.0f,df2));        
                double[] tmp = df1;
                df1 = df2;
                df2 = tmp;                         // swap derivatives
                d2 = scalar(df1,s);
                if (d2 > 0) {
                    s = svmult(-1.0f, df1);                              // otherwise use steepest direction
                    d2 = scalar(svmult(-1.0f,s), s);
                }
                z1 = z1 * Math.min(RATIO, d1/(d2-MINDOUBLE));          // slope ratio but max RATIO
                d1 = d2;
                ls_failed = false;                              // this line search did not fail   
            } 
            else {
                X = X0;
                setWeights(X);
                f1 = f0;
                df1 = df0;
                if (ls_failed || (iterations_i > Math.abs(length))) 
                    break;        // line search failed twice in a row, or we ran out of time, so we give up
                double[] tmp = df1;
                df1 = df2;
                df2 = tmp;                         // swap derivatives
                s = svmult(-1.0f, df1);                          // try steepest
                d1 = scalar(svmult(-1.0f, s), s);
                z1 = 1.0f/(1.0f-d1);          
                ls_failed = true;                                    // this line search failed
            }                   
        }
    }

}
