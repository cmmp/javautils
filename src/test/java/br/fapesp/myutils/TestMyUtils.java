package br.fapesp.myutils;

import static org.junit.Assert.*;

import org.junit.Test;

import weka.core.Instances;

public class TestMyUtils {
	
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
