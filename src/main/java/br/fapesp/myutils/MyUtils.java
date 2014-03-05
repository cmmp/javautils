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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import javax.swing.JComponent;
import javax.swing.JFrame;

import org.apache.commons.math3.util.MathArrays;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Utility class with many auxiliary methods.
 * 
 * @author cassio
 * 
 */
public class MyUtils {
	
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
		int N = data.length;		
		int[][] mst = new int[N-1][2];
		double[][] D = getEuclideanMatrix(data); // euclidean dissimilarity matrix
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
	 * Compute the Minimum Spanning Tree (MST) using Prim's algorithm - O(n^2).
	 * 
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
					ith = new Instance(m);
				else
					ith = new Instance(m + 1);
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

	public static void print_matrix(double[][] matrix) {
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
			for (int j = 0; j < ncols; j++)
				System.out.print(matrix[i][j] + " ");
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

	public static FastVector getUniqueElements(FastVector fv) {
		FastVector unique = new FastVector(fv.size());
		for (int i = 0; i < fv.size(); i++) {
			if (!unique.contains(fv.elementAt(i)))
				unique.addElement(fv.elementAt(i));
		}
		return unique;
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
