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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Scanner;

import javax.swing.JFrame;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.junit.Ignore;
import org.junit.Test;

import weka.core.Instances;

public class TestMyUtils {
	
	@Test
	public void testAdjustedRandIndex() {
		int[] got = new int[] {1, 2, 1, 1, 3, 2};
		int[] expect = new int[] {1, 1, 2, 2, 3, 3};
		
		double ari = MyUtils.computeAdjustedRandIndex(got, expect);
		
		assertEquals(0.07407407, ari, 1e-5);
	}
	
	@Test
	@Ignore
	public void testJFreeChart() {
		double[][] data = new double[][] { {1, 2}, {3, 4}, {5,6} };
		XYSeries xy = new XYSeries("Teste");
		xy.add(1, 2);
		xy.add(2,4);
		xy.add(3,6);
		
		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(xy);
		
		JFreeChart chart = ChartFactory.createScatterPlot("Scatter teste", "x", "y", dataset);
			
		ChartFrame frame =new ChartFrame("XYLine Chart",chart);
		frame.pack();
		frame.setVisible(true);
	}
	
	@Test
	public void testHausdorff() {
		double[][] x = { {0, 0}, {1,1} };
		double[][] y = { {3.0, -0.5}, {0.2, -3.0}, {3.1, -5.0} };
		
		double h = MyUtils.hausdorff(x, y);
		
		assertEquals(5.883026, h, 1e-5);
	}
	
	@Test
	@Ignore
	public void testEmbedd() {
		
		// R output for comparison:
//		> require(tseriesChaos)
//		Loading required package: tseriesChaos
//		Loading required package: deSolve
//		> x = c(1,5,10,20,-2,-3,4,8,10,15,23,25,40)
//		> x
//		 [1]  1  5 10 20 -2 -3  4  8 10 15 23 25 40
//		> length(x)
//		[1] 13
//		> embedd(x,
//		d=     lags=  m=     x=
//		> embedd(x, m = 3, d = 2)
//		      V1/0 V1/2 V1/4
//		 [1,]    1   10   -2
//		 [2,]    5   20   -3
//		 [3,]   10   -2    4
//		 [4,]   20   -3    8
//		 [5,]   -2    4   10
//		 [6,]   -3    8   15
//		 [7,]    4   10   23
//		 [8,]    8   15   25
//		 [9,]   10   23   40
		
		
		double[] series = { 1,5,10,20,-2,-3,4,8,10,15,23,25,40 };
		double[][] rec = MyUtils.embedd(series, 2, 3);
		MyUtils.print_matrix(rec);
	}
	
	@Test
	public void testRange() {
		int[] seq = MyUtils.range(1, 20, 3);
		int[] expected = {1, 4, 7, 10, 13, 16, 19};
		assertEquals(expected.length, seq.length);
		
		for(int i = 0; i < expected.length; i++)
			assertEquals(expected[i], seq[i]);
	}
	
	@Test
	public void testFastPrim() {
		double[][] data = new double[][] { {1., 2}, {3, 4}, {0, -1}, {3,6}, {7,8}, {9,10}, {-2,3}};
		int[][] mst = MyUtils.fastPrim(data);
		int[][] expected = new int[][] {{0,1}, {1,3}, {0,2}, {0,6}, {3,4}, {4,5}};
		for(int i = 0 ; i < expected.length; i++) {
			//System.out.printf("Expected: [%d][%d] got: [%d][%d]\n", expected[i][0], expected[i][1], mst[i][0], mst[i][1]);
			for(int j = 0; j < 2; j++)
				assertEquals(expected[i][j], mst[i][j]);
		}
	}
	
	@Test
	public void testMinimumSpanningTreePrim() {
		double[][] data = new double[][] { {1., 2}, {3, 4}, {0, -1}, {3,6}, {7,8}, {9,10}, {-2,3}};
		int[][] mst = MyUtils.computeMinimumSpanningTreePrim(data);
		int[][] expected = new int[][] {{0,1}, {1,3}, {0,2}, {0,6}, {3,4}, {4,5}};
		for(int i = 0 ; i < expected.length; i++) {
			for(int j = 0; j < 2; j++)
				assertEquals(expected[i][j], mst[i][j]);
		}
	}
	
	@Ignore
	@Test
	public void testWeightedSampling() {
		ArrayList<Integer> values = new ArrayList<Integer>();
		values.add(1); values.add(2); values.add(3); values.add(4); values.add(5);
		double[] weights = new double[] {0.1, 0.45, 0.03, 0.3, 0.12}; // correct sort idx: 2, 0, 4, 3, 1
		// plot a histogram of the following results to check probs are ok:
		for(int i = 0; i < 1000; i++)
			System.out.println(MyUtils.weightedValueRandomSelection(values, weights, null, System.nanoTime()));
		
	}
	
	@Test
	public void testGenGaussian() {
		Instances in = MyUtils.genGaussianDataset(new double[][] {{0,0,0},{10,10,20}}, new double[][] {{1,1,2},{1,1,3}}, 30, 1234, true, false);
		assertEquals(60, in.numInstances());
		assertEquals(3, in.numAttributes());
		in = MyUtils.genGaussianDataset(new double[][] {{0,0,0},{10,10,20}}, new double[][] {{1,1,2},{1,1,3}}, 30, 1234, true, true);
		assertEquals(60, in.numInstances());
		assertEquals(4, in.numAttributes());
	}

	@Test
	public void testLinspace() {
		double start = 1;
		double end = 2;
		int n = 10;
		double[] expected = new double[] {1.0, 1.11, 1.22, 1.33, 1.44, 1.55, 1.66, 1.77, 1.88, 2.0};
		double deltadiff = 1e-2;
		
		double[] got = MyUtils.linspace(start, end, n);
		
		assertArrayEquals(expected, got, deltadiff);
	}
	
	@Test
	public void testGenExp() {
		double base = 2.0;
		int nsteps = 5;
		double[] expected = new double[] {2.0, 4.0, 8.0, 16.0, 32.0};
		double[] got = MyUtils.genExpSeries(base, nsteps);
		double deltadiff = 1e-10;
		assertArrayEquals(expected, got, deltadiff);
	}

}
