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

import java.util.ArrayList;

import br.fapesp.myutils.MyUtils;

/**
 * MinHeap code based on the book "Projeto de algoritmos
 * com implementações em Java e C++. Nivio Ziviani".
 * 
 * @author cassio
 *
 */
public class MinHeap implements MinPriorityQueue {
	
	private HeapKey[] v;
	private int n;
	
	public MinHeap(HeapKey[] elements) {
		this.v = new HeapKey[elements.length + 1];
		this.v[0] = null;
		for(int i = 1; i < elements.length + 1; i++) this.v[i] = elements[i-1];
		this.n = this.v.length - 1; // because we won't use the first position in v 
	}
	
	public void updateKeys(int u, double[][] dist) {
		double w;
		
		for (int i = 1; i < this.n + 1; i++) {
			EdgeElement e = (EdgeElement) this.v[i];
			w = dist[e.u][u];
			if (e.weight > w) {
				e.v = u;
				e.weight = w;
			}	
		}
		
		this.build();
	}
	
	public HeapKey getMinFast() {
		HeapKey min;
		if (this.n < 1) throw new RuntimeException("Empty heap!");
		min = this.v[1];
		this.v[1] = this.v[this.n--]; // because we're going to shrink by the last element
		//rebuild(1, this.n); // we'll do this manually later
		return min;
	}
	
	public HeapKey getMin() {
		HeapKey min;
		if (this.n < 1) throw new RuntimeException("Empty heap!");
		min = this.v[1];
		this.v[1] = this.v[this.n--]; // because we're going to shrink by the last element
		rebuild(1, this.n);
		return min;
	}
	
	public int[] findKeysWithElement(int u) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		for (int i = 1; i < this.n + 1; i++)
			if (u == ((EdgeElement) this.v[i]).u)
				list.add(i);
		int[] ret = new int[list.size()];
		for(int i = 0; i < list.size(); i++) ret[i] = list.get(i);
		return ret;
	}
	
	public void decreaseKey(int i, HeapKey newKey) {
		HeapKey x = this.v[i];
		x.alterKey(newKey);
		while((i > 1) && (x.compareTo(this.v[i / 2]) < 0)) {
			this.v[i] = this.v[i / 2];
			i /= 2;
		}
		this.v[i] = x;
	}
	
	public void rebuild(int left, int right) {
		int j = 2 * left;
		HeapKey x = this.v[left];
		
		while(j <= right) {
			if((j < right) && (this.v[j].compareTo(this.v[j+1]) > 0))
				j++;
			if(x.compareTo(this.v[j]) <= 0) break;
			this.v[left] = this.v[j];
			left = j; 
			j = left * 2;
		}
		
		this.v[left] = x;
	}
	
	public void build() {
		int left = this.n / 2 + 1;
		while(left > 1) {
			left--;
			rebuild(left, this.n);
		}
	}
	
	public void print() {
		for(int i = 1; i < this.n + 1; i++)
			System.out.println(v[i] + " ");
		System.out.println();
	}
	
//	public static void main(String[] args) {
//		HeapKey[] c = new HeapKey[7];
//		c[0] = new EdgeElement(0,1,10.0);
//		c[1] = new EdgeElement(0,3,8.0);
//		c[2] = new EdgeElement(1,2,7.0);
//		c[3] = new EdgeElement(2,4,6.0);
//		c[4] = new EdgeElement(2,5,4.0);
//		c[5] = new EdgeElement(7,8,3.0);
//		c[6] = new EdgeElement(1,3,2.0);
//		MinHeap mh = new MinHeap(c);
//		mh.build();
//		mh.print();
////		System.out.println("obtaining elements:");
////		System.out.println(mh.getMin());
////		System.out.println(mh.getMin());
////		System.out.println(mh.getMin());
////		System.out.println(mh.getMin());
////		System.out.println(mh.getMin());
////		System.out.println(mh.getMin());
////		System.out.println(mh.getMin());
//		System.out.println("decreasing a key:");
//		mh.decreaseKey(3, new EdgeElement(7, 8, 1.5));
//		mh.print();
//		
//		// print key idxs:
//		int[] keyidxs = mh.findKeysWithElement(0);
//		MyUtils.print_array(keyidxs);
//	}

}
