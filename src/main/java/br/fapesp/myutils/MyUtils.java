package br.fapesp.myutils;

import java.awt.Color;
import java.awt.Graphics;
import java.util.Random;

import javax.swing.JComponent;
import javax.swing.JFrame;

import org.apache.commons.math3.util.MathArrays;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Utility class with many auxiliary methods.
 * 
 * @author cassio
 * 
 */
public class MyUtils {
	
	/**
	 * Generates a Gaussian data set with K clusters and m dimensions 
	 * @param centers K x m matrix
	 * @param sigmas K x m matrix
	 * @param pointsPerCluster number of points per cluster
	 * @param seed for the RNG
	 * @param randomize should the order of the instances be randomized?
	 * @param supervised should class label be present? if true, the class is the m+1 attribute
	 * 
	 * @return
	 */
	public static Instances genGaussianDataset(double[][] centers, double[][] sigmas, int pointsPerCluster, long seed, boolean randomize, boolean supervised) {
		Random r = new Random(seed);
		
		int K = centers.length; // number of clusters
		int m = centers[0].length; // number of dimensions
		
		FastVector atts = new FastVector(m);
		for(int i = 0; i < m; i++)
			atts.addElement(new Attribute("at" + i));
		
		if(supervised) {
			FastVector cls = new FastVector(K);
			for(int i = 0; i < K; i++) cls.addElement("Gauss-" + i);
			atts.addElement(new Attribute("Class", cls));
		}
		
		Instances data;
		if (supervised)
			data = new Instances(K + "-Gaussians-supervised", atts, K * pointsPerCluster);
		else
			data = new Instances(K + "-Gaussians", atts, K * pointsPerCluster);
		
		if(supervised)
			data.setClassIndex(m);
			
		Instance ith;
		
		for(int i = 0; i < K ; i++) {
			for(int j = 0; j < pointsPerCluster; j++) {
				if(!supervised)
					ith = new Instance(m);
				else
					ith = new Instance(m+1);
				ith.setDataset(data);
				for(int k = 0; k < m; k++) 
					ith.setValue(k, centers[i][k] + (r.nextGaussian() * sigmas[i][k]));
				if (supervised)
					ith.setValue(m, "Gauss-" + i);
				data.add(ith);
			}
		}
		
		// run randomization filter if desired
		if(randomize)
			data.randomize(r);
		
		return data;
	}
	
	/**
	 * Generates an integer sequence of the form (start, start+1, ..., start + nsteps - 1)
	 * @param start
	 * @param nsteps
	 * @return
	 */
	public static int[] genIntSeries(int start, int nsteps) {
		int[] series = new int[nsteps];
		series[0] = start;
		for(int i = 1; i < nsteps; i++)
			series[i] = series[i - 1] + 1;
		return series;
	}
	
	/**
	 * Generate an exponential series of the form: (base^1, ..., base^nsteps)
	 * @param base
	 * @param nsteps
	 * @return
	 */
	public static double[] genExpSeries(double base, int nsteps) {
		double[] series = new double[nsteps];
		for(int i = 0; i < nsteps; i++)
			series[i] = Math.pow(base, i+1);
		return series;
	}
	
	/**
	 * Creates a linearly spaced double array with n elements.
	 * @param start first element
	 * @param end last element
	 * @param n number of elements
	 * @return linearly spaced array
	 */
	public static double[] linspace(double start, double end, int n) {
		double[] ar = new double[n];
		double step = (end - start) / (n - 1);
		
		if(n < 3) {
			throw new RuntimeException("n must be >= 3");
		}
		
		ar[0] = start;
		ar[n - 1] = end;
		
		for(int i = 1; i < n - 1; i++)
			ar[i] = ar[i - 1] + step;
		
		return ar;
	}
	
	/**
	 * Find the minimum value in the column of a matrix
	 * @param mat
	 * @param col
	 * @return
	 */
	public static double getMinInCol(double[][] mat, int col) {
		double min = Double.MAX_VALUE;
		for(int i = 0; i < mat.length; i++)
			if (mat[i][col] < min)
				min = mat[i][col];
		return min;
	}
	
	/**
	 * Find the maximum value in the column of a matrix
	 * @param mat
	 * @param col
	 * @return
	 */
	public static double getMaxInCol(double[][] mat, int col) {
		double max = Double.MIN_VALUE;
		for(int i = 0; i < mat.length; i++)
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
	 * @param data
	 * @return
	 */
	public static double[][] getEuclideanMatrix(Instances data) {
		int N = data.numInstances();
		double[][] dist = new double[N][N];
		double d;
		// EuclideanDistance euc = new EuclideanDistance(data); // distance
		// calculator from weka
		// euc.setDontNormalize(true);

		for (int i = 0; i < N - 1; i++) {
			for (int j = i + 1; j < N; j++) {
				// d = euc.distance(data.instance(i), data.instance(j));
				d = MathArrays.distance(data.instance(i).toDoubleArray(), data
						.instance(j).toDoubleArray());
				dist[i][j] = d;
				dist[j][i] = d;
			}
		}

		return (dist);
	}

	public static void print_array(FastVector array) {
		System.out.print("[ ");
		for (int i = 0; i < array.size(); i++)
			System.out.print(array.elementAt(i) + ", ");
		System.out.println("]");
	}

	public static void print_array(double[] array) {
		System.out.print("[ ");
		for (int i = 0; i < array.length; i++)
			System.out.print(array[i] + ", ");
		System.out.println("]");
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

	public static void print_array(int[] array) {
		System.out.print("[ ");
		for (int i = 0; i < array.length; i++)
			System.out.print(array[i] + ", ");
		System.out.println("]");
	}

}
