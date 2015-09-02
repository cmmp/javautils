/* Copyright (C) 2014  Cássio M. M. Pereira

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

package br.fapesp.myutils;

import java.awt.Color;
import java.awt.Graphics;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import javax.swing.JComponent;
import javax.swing.JFrame;

import org.apache.commons.math3.util.MathArrays;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import au.com.bytecode.opencsv.CSVReader;

/**
 * Utility class with many auxiliary methods.
 * 
 * @author cassio
 * 
 */
public class MyUtils {
	
	public static double[] computePercentilesFromNbreaks(int nbreaks) {
		double[] percentiles = new double[nbreaks];
		double step = 100.0 / (nbreaks + 1);
		
		for (int i = 0; i < nbreaks; i++)
			percentiles[i] = step + i * step;
		return percentiles;
	}
	
	/** 
	 * Computes the Jaccard external clustering coefficient.
	 * 0 - worst matching.
	 * 1 - best clustering match.
	 * 
	 * @param labels obtained through clustering
	 * @param supervision external labeling
	 * @return index in the range [0,1]
	 */
	public static double jaccardIndex(int[] labels, int[] supervision) {
		double jaccard = 0;
		
		if (labels.length != supervision.length)
			throw new RuntimeException("labels and supervision must have the same length!");
		
		int N = labels.length;
		double M = N * (N - 1) / 2.0; // number of pairs of points
		
		double a = 0; // number of pairs that belong to the same class and same cluster
		double b = 0; // number of pairs that belong to the same class and diff clusters
		double c = 0; // number of pairs that belong to diff classes and the same cluster
		
		for (int i = 0 ; i < N - 1; i++) {
			for (int j = i + 1; j < N; j++) {
				if (labels[i] == labels[j] && supervision[i] == supervision[j])
					a++;
				else if(labels[i] != labels[j] && supervision[i] == supervision[j])
					b++;
				else if(labels[i] == labels[j] && supervision[i] != supervision[j])
					c++;
			}
		}
	
		jaccard = a / (a + b + c);
		
		return jaccard;
	}
	
	/**
	 * Computes the Adjusted Rand Index
	 * @param labels
	 * @param supervision
	 * @return
	 */
	public static double computeAdjustedRandIndex(int[] labels, int[] supervision) {
		double ARI = 0.0;
		int N = labels.length;
		
		if (labels.length != supervision.length)
			throw new RuntimeException("labels and supervision must have the same length!");
		
		double M = N * (N - 1) / 2.0; // number of pairs of points
		
		double a = 0; // number of pairs that belong to the same class and the same cluster
		double b = 0; // number of pairs that belong to the same class but different clusters
		double c = 0; // number of pairs that belong to different classes but the same cluster
		
		for (int i = 0; i < N - 1; i++) {
			for (int j = i + 1; j < N; j++) {
				if (supervision[i] == supervision[j] && labels[i] == labels[j])
					a++;
				if (supervision[i] == supervision[j] && labels[i] != labels[j])
					b++;
				if (supervision[i] != supervision[j] && labels[i] == labels[j])
					c++;
			}
		}
		
		ARI = (a - ((a+c)*(a+b) / M)) / ( (((a+c)+(a+b)) / 2.0) - ((a+c)*(a+b) / M) );
				
		return ARI;
	}
	
	/**
	 * Compute the following scores for a binary classification
	 * problem:
	 * 
	 * True positives;
	 * True negatives;
	 * False positives (Type I error);
	 * False negatives (Type II error);
	 * Recall (sensitivity or true positive rate);
	 * Specificity (true negative rate);
	 * Precision (positive predictive value);
	 * Negative predictive value;
	 * Fall-out (false positive rate);
	 * False discovery rate;
	 * Accuracy;
	 * F1 Score;
	 * Matthews correlation coefficient.
	 * 
	 * @see <a href="http://en.wikipedia.org/wiki/Precision_and_recall">Precision and recall</a>
	 * @param expected the list of expected classification results (in true or false possibilities)
	 * @param actual the classifier's predictions
	 * @param printResults whether to print the results to stdout
	 * @return the scores as an array
	 */
	public static double[] computeBinaryClassificationScores(
			List<Boolean> expected, List<Boolean> actual,
			boolean printResults) {
		
		double[] results = new double[13];
		double tp = 0, tn = 0, fp = 0, fn = 0;
		boolean ex, got;
		
		if (expected.size() != actual.size())
			throw new RuntimeException("Expected and actual lists have a different size!");
		
		for (int i = 0; i < expected.size(); i++) {
			ex = expected.get(i); got = actual.get(i); 
			
			if (ex == false && ex == got)
				tn++;
			else if (ex == false)
				fp++;
			else if (ex == true && ex == got)
				tp++;
			else
				fn++;
		}
		
		results[0] = tp; results[1] = tn; results[2] = fp; results[3] = fn;
		results[4] = tp / (tp + fn); // recall
		results[5] = tn / (fp + tn); // specificity
		results[6] = tp / (tp + fp); // precision
		results[7] = tn / (tn + fn); // negative predictive value
		results[8] = fp / (fp + tn); // false positive rate
		results[9] = fp / (fp + tp); // false discovery rate
		results[10] = (tp + tn) / expected.size(); // accuracy
		results[11] = (2.0 * tp) / (2.0 * tp + fp + fn); // f1 score
		results[12] = (tp * tn - fp * fn) / Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); // matthews
		
		if (printResults) {
			printRep('-', 80);
			String prec = "%.2f";
			System.out.println("Score results:");
			System.out.println("True positives: " + tp);
			System.out.println("True negatives: " + tn);
			System.out.println("False positives: " + fp);
			System.out.println("False negatives: " + fn);
			System.out.printf("Recall (or sensitivity or true positive rate): " + prec + "\n", results[4]);
			System.out.printf("Specificity (true negative rate): " + prec + "\n",  results[5]);
			System.out.printf("Precision (positive predictive value): " + prec + "\n", results[6]);
			System.out.printf("Negative predictive value: " + prec + "\n",  results[7]);
			System.out.printf("Fall-out (false positive rate): " + prec + "\n",  results[8]);
			System.out.printf("False discovery rate: " + prec + "\n",  results[9]);
			System.out.printf("accuracy: " + prec + "\n",  results[10]);
			System.out.printf("F-1 score: " + prec + "\n",  results[11]);
			System.out.printf("Matthews correlation coefficient (+1 perfect, 0 random, -1 total disagreement): " + prec + "\n",  results[12]);
			printRep('-', 80);
		}
		
		return results;
	}
	
	private static double internalHausdorffAlg(double[][] x, double[][] y) {
		double h = 0;
		double dij, shortest;
		
		for (int i = 0; i < x.length; i++) {
			shortest = Double.MAX_VALUE;
			for (int j = 0; j < y.length; j++) {
				dij = squaredEuclideanDistance(x[i], y[j]);
				if (dij < shortest)
					shortest = dij;
			}
			if (shortest > h)
				h = shortest;
		}
		return Math.sqrt(h);
	}
	
	/**
	 * Computes the Hausdorff distance between the points in x and y.
	 * @see <a href="http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html">Hausdorff distance</a>
	 * @param x first input matrix
	 * @param y second input matrix
	 * @return Hausdorff distance between x and y
	 */
	public static double hausdorff(double[][] x, double[][] y) {	
		return Math.max(internalHausdorffAlg(x, y), internalHausdorffAlg(y, x));
	}
	
	/**
	 * Creates a sequence of elements from start to end (omited) with the step given.
	 * @param start initial element
	 * @param end end point (omitted!)
	 * @param step step size between elements
	 * @return the sequence
	 */
	public static int[] range(int start, int end, int step) {
		if (end <= start) {
			throw new RuntimeException("end must be greater than start!");
		}
		int n = (int) Math.ceil((end - start) / ((float) step));
		int[] seq = new int[n];
		
		seq[0] = start;
		
		for(int i = 1; i < n; i++)
			seq[i] = seq[i - 1] + step; 
		
		return seq;
	}
	
	/**
	 * Create the embedding of a series in phase space.
	 * @param series the original time series
	 * @param delay the separation dimension
	 * @param embedding the embedding dimension
	 * @return reconstructed series in phase space
	 */
	public static double[][] embedd(double[] series, int delay, int embedding) {
		int n = series.length;
		int m = n - (embedding - 1) * delay;
		double[][] rec = new double[m][embedding];
		int[] idxs;
		
		for (int i = 0; i < m; i++) {
			idxs = MyUtils.range(i, i + embedding * delay, delay);
			for (int j = 0; j < embedding; j++)
				rec[i][j] = series[idxs[j]];
		}
		
		return rec;
	}

	/**
	 * Estimate the maximum distance among points without computing the full
	 * Euclidean matrix. The method uses a subsample of the points to make
	 * the estimation.
	 * 
	 * @param points original data matrix (NOT a distance matrix).
	 * @param ratio the ratio as in (original number of points) / (ratio) to use as the number of samples
	 * @param factor the estimation will use (max dist in subsample) * (factor) as the estimate - use 1.0 for the sample estimate.
	 * @param rng random number generator object (may be null)
	 * @param seed seed for the RNG if no rng was passed as a parameter
	 * @return the estimate for the maximum distance
	 */
	public static double estimateMaxDist(double[][] points, int ratio, double factor, Random rng, long seed) {
		
		Random r;
		
		if (rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		int nsamples = points.length / ratio;
		
		if (nsamples <= 1) // we're probably dealing with a small matrix.
			nsamples = points.length;
		
		int ndim = points[0].length;
		
		double[][] sampled = new double[nsamples][ndim];
		
		int[] available = MyUtils.genIntSeries(0, points.length);
		int[] pos = MyUtils.sampleWithoutReplacement(available, nsamples, r, -1);
		
		for(int i = 0; i < nsamples; i++)
			System.arraycopy(points[pos[i]], 0, sampled[i], 0, ndim);
		
		double[][] dist = MyUtils.getEuclideanMatrix(sampled);
		
		return MyUtils.getMatrixMax(dist) * factor;
	}

    public static int argMax(ArrayList<Double> array) {
        int argmax = 0;
        double max = array.get(0);
        double v;

        for (int i = 1; i < array.size(); i++) {
            v = array.get(i);
            if (v > max) {
                max = v;
                argmax = i;
            }
        }

        return argmax;
    }
	
	/**
	 * Compute the squared euclidean distance between two doubles arrays
	 * @param x
	 * @param y
	 * @return the *squared* euclidean distance
	 */
	public static double squaredEuclideanDistance(double[] x, double[] y) {
		double dist = 0;
		double diff;
		
		for (int i = 0; i < x.length; i++) {
			diff = x[i] - y[i];
			dist += diff * diff;
		}
		
		return dist;
	}

    /**
     * Convert a list to an array
     * @param list
     * @return
     */
    public static int[] listToArrayInt(List<Integer> list) {
        int[] array = new int[list.size()];
        for (int i = 0; i < list.size(); i++)
            array[i] = list.get(i);
        return array;
    }

    /**
	 * Convert a list to an array
	 * @param list
	 * @return
	 */
	public static double[] listToArray(List<Double> list) {
		double[] array = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
			array[i] = list.get(i);
		return array;
	}
	
	/**
	 * Get the most frequently occurring number in a sequence
	 * @param values the sequence of numbers
	 * @return the number that appears the most times in the sequence
	 */
	public static double getMostFrequenceOccurrence(double[] values) {
		HashMap<Double, Integer> mapCount = new HashMap<Double, Integer>();
		int mostFreq = -1;
		double mostFreqVal = -1;
		
		for (int i = 0; i < values.length; i++) {
			if (!mapCount.containsKey(values[i]))
				mapCount.put(values[i], 0);
			else
				mapCount.put(values[i], mapCount.get(values[i]) + 1);
		}
		for (Double key : mapCount.keySet()) {
			if (mapCount.get(key) > mostFreq) {
				mostFreq = mapCount.get(key);
				mostFreqVal = key;
			}
		}
		return mostFreqVal;
	}
	
	/**
	 * Print a character n times on stdout
	 * @param c character
	 * @param ntimes number of times to print
	 */
	public static void printRep(char c, int ntimes) {
		StringBuffer sb = new StringBuffer(c);
		for (int i = 0; i < ntimes - 1; i++)
			sb.append(c);
		System.out.println(sb);
	}
	
	/**
	 * Print the current system timestamp to stdout
	 */
	public static void printTimestamp() {
		System.out.println(getCurrentTimestamp());
	}
	
	
	/**
	 * Get current timestamp of the system
	 * @return timestamp with the format: dow mon dd hh:mm:ss zzz yyyy
	 */
	public static String getCurrentTimestamp() {
		Date d = new Date();
		return d.toString();
	}
	
	/**
	 * Read a CSV file to a double's matrix.
	 * 
	 * @param filePath path to the csv file
	 * @param hasHeader whether the first line is a header
	 * @param separator the character separating each element in a line
	 * @return a double's matrix representation of the data set
	 */
	public static double[][] readCSVdataSet(String filePath, boolean hasHeader, char separator) {
		double[][] data = null;
		int N, m;
		
		try {
			CSVReader reader = new CSVReader(new FileReader(filePath), separator);
			
			List<String[]> lines = reader.readAll();
			
			if (hasHeader)
				lines.remove(0);
			
			N = lines.size();
			m = lines.get(0).length;
					
			data = new double[N][m];
			
			for (int i = 0; i < N; i++) 
				for (int j = 0; j < m; j++)
					data[i][j] = Double.parseDouble(lines.get(i)[j]);
			
			lines = null;
			
			reader.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return data;
	}
	
	/**
	 * Rescale data to be in the range newMin:newMax
	 * @param data the data set
	 * @param newMin new minimum value
	 * @param newMax new maximum value
	 * @return the rescaled data
	 */
	public static double[] rescaleToRange(double[] data, double newMin, double newMax) {
		double[] rescaled = new double[data.length];
		double oldMin = getMin(data);
		double oldMax = getMax(data);
		
		for (int i = 0; i < data.length; i++)
			rescaled[i] = (((data[i] - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin;
		
		return rescaled;
	}
	
	/**
	 * A faster version of Prim's algorithm using a priority queue to 
	 * compute the Minimum Spanning Tree (MST).
	 * 
	 * @param data points in R^n
	 * @param D euclidean distance matrix already computed
	 * @return a 2-d array listing the edges of the MST.
	 */
	public static int[][] fastPrim(double[][] data, double[][] D) {	
		int N = data.length;		
		int[][] mst = new int[N-1][2];
		HeapKey[] keys = new HeapKey[N-1];
		
		for (int i = 1; i < N; i++)
			keys[i - 1] = new EdgeElement(i, 0, D[0][i]);
		
		MinHeap mh = new MinHeap(keys);
		mh.build();
		
		for (int i = 0; i < N - 1; i++) {
			EdgeElement e = (EdgeElement) mh.getMinFast();
			mst[i][0] = e.v;
			mst[i][1] = e.u;
			mh.updateKeys(e.u, D);
		}
			
		return mst;
	}
	
	/**
	 * A faster version of Prim's algorithm using a priority queue to 
	 * compute the Minimum Spanning Tree (MST).
	 * 
	 * @param data points in R^n
	 * @return a 2-d array listing the edges of the MST.
	 */
	public static int[][] fastPrim(double[][] data) {	
		double[][] D = getEuclideanMatrix(data); // euclidean dissimilarity matrix
		return(fastPrim(data, D));
	}
	
	/**
	 * Compute the Minimum Spanning Tree (MST) using Prim's algorithm - O(n^2).
	 * 
	 * @deprecated
	 * @param data points in R^n
	 * @return a 2-d array listing the edges of the MST.
	 */
	public static int[][] computeMinimumSpanningTreePrim(double[][] data) {	
		int N = data.length;		
		int[][] mst = new int[N-1][2];
		double[][] D = getEuclideanMatrix(data); // euclidean dissimilarity matrix
		double minw;
		int ichosen = -1, v_to_add = -1;
		int k = 0;
		
		HashSet<Integer> vin = new HashSet<Integer>();
		HashSet<Integer> vout = new HashSet<Integer>();
		
		for(int i = 1; i < N; i++) 
			vout.add(i); // put all vertices on the out set, except the first
		
		vin.add(0); // add the first vertex
		
		while (vout.size() > 0) {
			// find the edge with minimum weight to add:
			minw = Double.MAX_VALUE;
			for(int i : vin )
				for(int j : vout) {
					if(D[i][j] < minw) {
						ichosen = i;
						minw = D[i][j];
						v_to_add = j;
					}
				}
			vout.remove(v_to_add);
			vin.add(v_to_add);
			mst[k][0] = ichosen; mst[k][1] = v_to_add;
			k++;
		}
		
		return mst;
	}
	
	public static String arrayToString(int[] ar) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(int i = 0; i < ar.length; i++) {
			sb.append(ar[i]);
			if(i + 1 < ar.length)
				sb.append(", ");
		}
		sb.append("]");
		return sb.toString();
	}
	
	public static String arrayToString(double[] ar) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(int i = 0; i < ar.length; i++) {
			sb.append(ar[i]);
			if(i + 1 < ar.length)
				sb.append(", ");
		}
		sb.append("]");
		return sb.toString();
	}
	
	public static String arrayToString(String[] ar) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(int i = 0; i < ar.length; i++) {
			sb.append(ar[i]);
			if(i + 1 < ar.length)
				sb.append(", ");
		}
		sb.append("]");
		return sb.toString();
	}
	
	public static String arrayToString(boolean[] ar) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(int i = 0; i < ar.length; i++) {
			sb.append(ar[i]);
			if(i + 1 < ar.length)
				sb.append(", ");
		}
		sb.append("]");
		return sb.toString();
	}
	
	public static void print_dataset_as_matrix(Instances data) {
		for(int i = 0; i < data.numInstances(); i++) {
			for(int j = 0; j < data.numAttributes(); j++)
				System.out.print(data.instance(i).value(j) + " ");
			System.out.println();
		}
	}
	
	public static ArrayList<Integer> toArrayList(int[] values) {
		ArrayList<Integer> ar = new ArrayList<Integer>();
		for(int i = 0; i < values.length; i++) ar.add(values[i]);
		return ar;
	}
	
	public static ArrayList<Double> toArrayList(double[] values) {
		ArrayList<Double> ar = new ArrayList<Double>();
		for(int i = 0; i < values.length; i++) ar.add(values[i]);
		return ar;
	}
	
	/**
	 * Selecting a value from values array at random using the weights provided
	 * @param values
	 * @param weights MUST sum to 1.0
	 * @param rng
	 * @param seed
	 * @return
	 */
	public static int weightedValueRandomSelection(ArrayList<Integer> values, double[] weights, Random rng, long seed) {
		if(Math.abs(MyUtils.sum(weights) - 1.0) > 0.01)
			throw new RuntimeException("weights must sum to 1.0!");
		if(values.size() != weights.length)
			throw new RuntimeException("values and weights must be of the same size!");
		
		Random r;
		
		if(rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		double[] _weights = new double[weights.length];
		int[] sortidx = new int[weights.length];
		double flip;
		
		System.arraycopy(weights, 0, _weights, 0, weights.length);
		
		Arrays.sort(_weights);
		
		ArrayList<Integer> _values = new ArrayList<Integer>();
		
		// find out the indexes of the correct sorting:
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights.length; j++)
				if(_weights[i] == weights[j]) {
					sortidx[i] = j;
					break;
				}
		}
		
		// add the values in the sort order:
		for(int i = 0; i < weights.length; i++)
			_values.add(values.get(sortidx[i]));
		
		flip = r.nextDouble();
		
		// now check from the largest probability to the lowest, which one was selected:
		double cumsum = 0;
		
		for(int i = weights.length - 1; i >= 0; i--) {
			cumsum += _weights[i];
			if (flip <= cumsum)
				return _values.get(i);
		}
		
		throw new RuntimeException("error occurred");
	}
	
	/** 
	 * Flip a coin using provided weights.
	 * @param probTrue the probability the event is true 
	 * @param rng
	 * @param seed
	 * @return
	 */
	public static boolean weightedCoinFlip(double probTrue, Random rng, long seed) {
		Random r;
		double flip;
		if (rng == null)
			r = new Random(seed);
		else
			r = rng;
		flip = r.nextDouble();
		if (flip <= probTrue)
			return true;
		else
			return false;
	}
	
	public static int[] createArrayFromHashSet(HashSet<Integer> hash) {
		int[] ar = new int[hash.size()];
		int i = 0;
		for(Integer val : hash) {
			ar[i++] = val;
		}
		return ar;
	}
	
	public static HashSet<Integer> createHashSetFromArray(int[] array) {
		HashSet<Integer> hash = new HashSet<Integer>();
		for(int i = 0; i < array.length; i++) hash.add(array[i]);
		return hash;
	}
	
	public static double[] uniformlySample(double low, double high, int nsamples, Random rng, long seed) {
		double[] samples = new double[nsamples];
		Random r;
		
		if (rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		for(int i = 0; i < nsamples; i++)
			samples[i] = r.nextDouble() * (high - low) + low;
		
		return samples;
	}
	
	/**
	 * Create an array of val repeated n times
	 * @param val
	 * @param n
	 * @return
	 */
	public static int[] repeat(int val, int n) {
		int[] x = new int[n];
		for(int i = 0; i < n; i++)
			x[i] = val;
		return x;
	}
	
	/**
	 * Compute the sum of the array x
	 * @param x
	 * @return
	 */
	public static double sum(double[] x) {
		double sum = 0;
		for(int i = 0; i < x.length; i++)
			sum += x[i];
		return sum;
	}
	
	/**
	 * Compute the sum of the array x
	 * @param x
	 * @return
	 */
	public static int sum(int[] x) {
		int sum = 0;
		for(int i = 0; i < x.length; i++)
			sum += x[i];
		return sum;
	}
	
	/**
	 * Returns nsamples from values, chosen randomly _without_ replacement.
	 * @param values
	 * @param nsamples
	 * @param rng optional random number generator
	 * @param seed if no rng is passed, create one with this seed
	 * @return
	 */
	public static double[] sampleWithoutReplacement(double[] values, int nsamples, Random rng, long seed) {
		if (nsamples > values.length)
			throw new RuntimeException("Requested sampling of more values than are available in array!");
		Random r;
		if (rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		double[] samples = new double[nsamples];
		ArrayList<Double> al = new ArrayList<Double>();
		
		for(int i = 0; i < values.length; i++) al.add(values[i]);
		
		for(int i = 0; i < nsamples; i++) 
			samples[i] = al.remove(r.nextInt(al.size()));
		
		return samples;
	}
	
	/**
	 * Returns nsamples from values, chosen randomly _without_ replacement.
	 * @param values
	 * @param nsamples
	 * @param rng optional random number generator
	 * @param seed if no rng is passed, create one with this seed
	 * @return
	 */
	public static int[] sampleWithoutReplacement(int[] values, int nsamples, Random rng, long seed) {
		if (nsamples > values.length)
			throw new RuntimeException("Requested sampling of more values than are available in array!");
		Random r;
		if (rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		int[] samples = new int[nsamples];
		ArrayList<Integer> al = new ArrayList<Integer>();
		
		for(int i = 0; i < values.length; i++) al.add(values[i]);
		
		for(int i = 0; i < nsamples; i++) 
			samples[i] = al.remove(r.nextInt(al.size()));
		
		return samples;
	}
	
	/**
	 * Returns nsamples from values, chosen randomly _with_ replacement.
	 * @param values
	 * @param nsamples
	 * @param rng optional random number generator
	 * @param seed if no rng is passed, create one with this seed
	 * @return
	 */
	public static double[] sampleWithReplacement(double[] values, int nsamples, Random rng, long seed) {
		Random r;
		double[] samples = new double[nsamples];
		
		if(rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		for(int i = 0; i < nsamples; i++) {
			samples[i] = values[r.nextInt(values.length)];
		}
		
		return samples;
	}
	
	/**
	 * Returns nsamples from values, chosen randomly _with_ replacement.
	 * @param values
	 * @param nsamples
	 * @param rng optional random number generator
	 * @param seed if no rng is passed, create one with this seed
	 * @return
	 */
	public static int[] sampleWithReplacement(int[] values, int nsamples, Random rng, long seed) {
		Random r;
		int[] samples = new int[nsamples];
		
		if(rng == null)
			r = new Random(seed);
		else
			r = rng;
		
		for(int i = 0; i < nsamples; i++) {
			samples[i] = values[r.nextInt(values.length)];
		}
		
		return samples;
	}
	

	/**
	 * Remove supervision from a data set
	 * 
	 * @param dataset
	 *            an Instances data set
	 * @param classIndex
	 *            if -1 use the last attribute
	 * @return a new copy of the data set with supervision removed
	 */
	public static Instances removeSupervision(Instances dataset, int classIndex) {
		Instances unsupervised = new Instances(dataset);
		String index = classIndex == -1 ? "last" : String.valueOf(classIndex);
		
		// Use the Remove filter to delete supervision from the data set:
		Remove rm = new Remove();
		rm.setAttributeIndices(index);

		try {
			rm.setInputFormat(unsupervised);
			unsupervised = Filter.useFilter(unsupervised, rm);
			return unsupervised;
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null;
	}

	public static Instances genGaussianDatasetWithSigmaEvolution(
			double[][] centers, double[][] sigmas, double[][] sigmas2,
			int pointsPerCluster, long seed, boolean randomize) {
		Instances dataset1 = genGaussianDataset(centers, sigmas,
				pointsPerCluster, seed, randomize, false);
		Instances dataset2 = genGaussianDataset(centers, sigmas2,
				pointsPerCluster, seed + 59387, randomize, false);

		for (int i = 0; i < dataset2.numInstances(); i++)
			dataset1.add(dataset2.instance(i));

		return dataset1;
	}

	/**
	 * Generates a Gaussian data set with K clusters and m dimensions
	 * 
	 * @param centers
	 *            K x m matrix
	 * @param sigmas
	 *            K x m matrix
	 * @param pointsPerCluster
	 *            number of points per cluster
	 * @param seed
	 *            for the RNG
	 * @param randomize
	 *            should the order of the instances be randomized?
	 * @param supervised
	 *            should class label be present? if true, the class is the m+1
	 *            attribute
	 * 
	 * @return
	 */
	public static Instances genGaussianDataset(double[][] centers,
			double[][] sigmas, int pointsPerCluster, long seed,
			boolean randomize, boolean supervised) {
		Random r = new Random(seed);

		int K = centers.length; // number of clusters
		int m = centers[0].length; // number of dimensions

		FastVector atts = new FastVector(m);
		for (int i = 0; i < m; i++)
			atts.addElement(new Attribute("at" + i));

		if (supervised) {
			FastVector cls = new FastVector(K);
			for (int i = 0; i < K; i++)
				cls.addElement("Gauss-" + i);
			atts.addElement(new Attribute("Class", cls));
		}

		Instances data;
		if (supervised)
			data = new Instances(K + "-Gaussians-supervised", atts, K
					* pointsPerCluster);
		else
			data = new Instances(K + "-Gaussians", atts, K * pointsPerCluster);

		if (supervised)
			data.setClassIndex(m);

		Instance ith;

		for (int i = 0; i < K; i++) {
			for (int j = 0; j < pointsPerCluster; j++) {
				if (!supervised)
					ith = new DenseInstance(m);
				else
					ith = new DenseInstance(m + 1);
				ith.setDataset(data);
				for (int k = 0; k < m; k++)
					ith.setValue(k, centers[i][k]
							+ (r.nextGaussian() * sigmas[i][k]));
				if (supervised)
					ith.setValue(m, "Gauss-" + i);
				data.add(ith);
			}
		}

		// run randomization filter if desired
		if (randomize)
			data.randomize(r);

		return data;
	}

	/**
	 * Generates an integer sequence of the form (start, start+1, ..., start +
	 * nsteps - 1)
	 * 
	 * @param start
	 * @param nsteps
	 * @return
	 */
	public static int[] genIntSeries(int start, int nsteps) {
		int[] series = new int[nsteps];
		series[0] = start;
		for (int i = 1; i < nsteps; i++)
			series[i] = series[i - 1] + 1;
		return series;
	}

	/**
	 * Generate an exponential series of the form: (base^1, ..., base^nsteps)
	 * 
	 * @param base
	 * @param nsteps
	 * @return
	 */
	public static double[] genExpSeries(double base, int nsteps) {
		double[] series = new double[nsteps];
		for (int i = 0; i < nsteps; i++)
			series[i] = Math.pow(base, i + 1);
		return series;
	}
	
	/**
	 * Creates an exponentially increasing sequence
	 * @param start
	 * @param end
	 * @param n number of elements
	 * @return An n-element sequence of exponentially increasing elements
	 */
	public static double[] expspace(double start, double end, int n) {
		double[] ar = new double[n];
		double step = (Math.log10(end) - Math.log10(start)) / (n - 1);

		if (n < 3) {
			throw new RuntimeException("n must be >= 3");
		}

		ar[0] = start;
		ar[n - 1] = end;

		for (int i = 1; i < n - 1; i++)
			ar[i] = Math.exp(Math.log(10) * (Math.log10(ar[i - 1]) + step));

		return ar;
	}

	/**
	 * Creates a linearly spaced double array with n elements.
	 * 
	 * @param start
	 *            first element
	 * @param end
	 *            last element
	 * @param n
	 *            number of elements
	 * @return linearly spaced array
	 */
	public static double[] linspace(double start, double end, int n) {
		double[] ar = new double[n];
		double step = (end - start) / (n - 1);

		if (n < 3) {
			throw new RuntimeException("n must be >= 3");
		}

		ar[0] = start;
		ar[n - 1] = end;

		for (int i = 1; i < n - 1; i++)
			ar[i] = ar[i - 1] + step;

		return ar;
	}

	/**
	 * Find the minimum value in the column of a matrix
	 * 
	 * @param mat
	 * @param col
	 * @return
	 */
	public static double getMinInCol(double[][] mat, int col) {
		double min = Double.MAX_VALUE;
		for (int i = 0; i < mat.length; i++)
			if (mat[i][col] < min)
				min = mat[i][col];
		return min;
	}

	/**
	 * Find the maximum value in the column of a matrix
	 * 
	 * @param mat
	 * @param col
	 * @return
	 */
	public static double getMaxInCol(double[][] mat, int col) {
		double max = Double.MIN_VALUE;
		for (int i = 0; i < mat.length; i++)
			if (mat[i][col] > max)
				max = mat[i][col];
		return max;
	}

	/**
	 * Plot a matrix consisting of 0's and 1's.
	 * 
	 * @param map
	 */
	public static void plotBinaryMatrix(final int[][] map, int width, int height) {
		JFrame frame = new JFrame("Binary matrix plot");

		frame.add(new JComponent() {
			private static final long serialVersionUID = 1L;

			@Override
			protected void paintComponent(Graphics g) {
				super.paintComponent(g);

				int w = getWidth() / map.length;
				int h = getHeight() / map[0].length;

				for (int i = 0; i < map.length; i++) {
					for (int j = 0; j < map[0].length; j++) {
						if (map[i][j] != 0) {
							g.setColor(new Color(map[i][j]));
							g.fillRect(i * w, j * h, w, h);
						}
					}
				}
			}
		});

		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(width, height);
		frame.setVisible(true);
	}
	
	/**
	 * Compute the Euclidean dissimilarity matrix on data
	 * 
	 * @param N data points in R^n
	 * @return an NxN matrix containing the Euclidean distance among points.
	 */
	public static double[][] getEuclideanMatrix(double[][] data) {
		int N = data.length;
		double[][] dist = new double[N][N];
		double d;

		for (int i = 0; i < N - 1; i++) {
			for (int j = i + 1; j < N; j++) {
				d = MathArrays.distance(data[i], data[j]);
				dist[i][j] = d;
				dist[j][i] = d;
			}
		}

		return (dist);
	}
	
	/**
	 * Convert an Instances data set to a doubles matrix.
	 * @param data
	 * @return data as a double array
	 */
	public static double[][] convertInstancesToDoubleMatrix(Instances data) {
		int N = data.numInstances();
		int m = data.numAttributes();
		double[][] ddata = new double[N][m];
		double[] temp;
		
		for (int i = 0; i < N; i++) {
			temp = data.instance(i).toDoubleArray();
			for (int j = 0; j < m; j++)
				ddata[i][j] = temp[j];
		}
		
		return(ddata);
	}

	/**
	 * Compute the Euclidean dissimilarity matrix on data
	 * 
	 * @param data
	 * @return
	 */
	public static double[][] getEuclideanMatrix(Instances data) {
		return (getEuclideanMatrix(convertInstancesToDoubleMatrix(data)));
	}
	
	public static void print_array(String[] array) {
		System.out.println(arrayToString(array));
	}

	public static void print_array(FastVector array) {
		System.out.print("[");
		for (int i = 0; i < array.size(); i++) {
			System.out.print(array.elementAt(i));
			if(i + 1 < array.size())
				System.out.print(", ");
		}
		System.out.println("]");
	}
	
	public static void print_array(boolean[] array) {
		System.out.println(arrayToString(array));
	}

	public static void print_array(double[] array) {
		System.out.println(arrayToString(array));
	}
	
	/**
	 * Print matrix with a nice formatting
	 * @param matrix mat to be printed
	 * @param fmt for instance, %.2f
	 */
	public static void print_matrix(double[][] matrix, String fmt) {
		int nrows, ncols;
		ncols = matrix[0].length;
		nrows = matrix.length;

		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				System.out.format(fmt, matrix[i][j]);
				if(j + 1 < ncols)
					System.out.print(" ");
			}
			System.out.print("\n");
		}

	}


	public static void print_matrix(double[][] matrix) {
		int nrows, ncols;
		ncols = matrix[0].length;
		nrows = matrix.length;

		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				System.out.print(matrix[i][j]);
				if(j + 1 < ncols)
					System.out.print(" ");
			}
			System.out.print("\n");
		}

	}

	/**
	 * Find the index of the smallest element in fv. Assumes fv is a list of
	 * doubles.
	 * 
	 * @param fv
	 * @return
	 */
	public static int argMin(FastVector fv) {
		double min = Double.MAX_VALUE;
		int idx = -1;
		for (int i = 0; i < fv.size(); i++) {
			if ((Double) fv.elementAt(i) < min) {
				min = (Double) fv.elementAt(i);
				idx = i;
			}
		}
		return idx;
	}

	public static void print_matrix(int[][] matrix) {
		int nrows, ncols;
		ncols = matrix[0].length;
		nrows = matrix.length;

		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++) {
				System.out.print(matrix[i][j]);
				if(j + 1 < ncols)
					System.out.print(" ");
			}
			System.out.print("\n");
		}
	}

	public static void print_matrix(boolean[][] matrix) {
		int nrows, ncols;
		ncols = matrix[0].length;
		nrows = matrix.length;

		for (int i = 0; i < nrows; i++) {
			for (int j = 0; j < ncols; j++)
				System.out.print(matrix[i][j] + " ");
			System.out.print("\n");
		}
	}
	
	/**
	 * Get the unique elements in a sequence of values
	 * @param values a sequence of numbers
	 * @return a hashset containing each unique occurence
	 */
	public static HashSet<String> getUniqueElements(String[] values) {
		HashSet<String> unique = new HashSet<String>();
		
		for (int i = 0; i < values.length; i++) 
			unique.add(values[i]);
		
		return unique;
	}
	
	/**
	 * Get the unique elements in a sequence of values
	 * @param values a sequence of numbers
	 * @return a hashset containing each unique occurence
	 */
	public static HashSet<Integer> getUniqueElements(int[] values) {
		HashSet<Integer> unique = new HashSet<Integer>();
		
		for (int i = 0; i < values.length; i++) 
			unique.add(values[i]);
		
		return unique;
	}
	
	/**
	 * Get the unique elements in a sequence of values
	 * @param values a sequence of numbers
	 * @return a hashset containing each unique occurence
	 */
	public static HashSet<Double> getUniqueElements(double[] values) {
		HashSet<Double> unique = new HashSet<Double>();
		
		for (int i = 0; i < values.length; i++) 
			unique.add(values[i]);
		
		return unique;
	}

	public static FastVector getUniqueElements(FastVector fv) {
		FastVector unique = new FastVector(fv.size());
		for (int i = 0; i < fv.size(); i++) {
			if (!unique.contains(fv.elementAt(i)))
				unique.addElement(fv.elementAt(i));
		}
		return unique;
	}
	
	public static double getMatrixMax(double[][] matrix) {
		double max = matrix[0][0];
		for(int i = 0; i < matrix.length; i++)
			for(int j = 0; j < matrix[0].length; j++)
				if (matrix[i][j] > max)
					max = matrix[i][j];
		return max;
	}
	
	public static double getMatrixMin(double[][] matrix) {
		double min = matrix[0][0];
		for(int i = 0; i < matrix.length; i++)
			for(int j = 0; j < matrix[0].length; j++)
				if (matrix[i][j] < min)
					min = matrix[i][j];
		return min;
	}

	public static int getMin(int[] vec) {
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < vec.length; i++)
			if (vec[i] < min)
				min = vec[i];
		return min;
	}

	public static int getMax(int[] vec) {
		int max = Integer.MIN_VALUE;
		for (int i = 0; i < vec.length; i++)
			if (vec[i] > max)
				max = vec[i];
		return max;
	}
	
	public static double getMin(double[] vec) {
		double min = Double.MAX_VALUE;
		for (int i = 0; i < vec.length; i++)
			if (vec[i] < min)
				min = vec[i];
		return min;
	}

	public static double getMax(double[] vec) {
		double max = Double.MIN_VALUE;
		for (int i = 0; i < vec.length; i++)
			if (vec[i] > max)
				max = vec[i];
		return max;
	}

	public static void print_array(int[] array) {
		System.out.println(arrayToString(array));
	}

}
