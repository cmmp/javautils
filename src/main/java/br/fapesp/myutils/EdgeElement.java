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

public class EdgeElement implements HeapKey {
	
	public int u;
	public int v;
	public double weight;
	
	public EdgeElement(int u, int v, double weight) {
		this.u = u;
		this.v = v;
		this.weight = weight;
	}
	
	@Override
	public String toString() {
		return "Edge element (" + u + "," + v + "): " + "weight = " + weight;
	}

	@Override
	public int compareTo(HeapKey o) {
		if (!(o instanceof EdgeElement))
			throw new RuntimeException("Comparing to another object type!");
		EdgeElement e = (EdgeElement) o;
		if (weight < e.weight) return -1;
		if (weight > e.weight) return 1;
		
		return 0;
	}

	@Override
	public void alterKey(HeapKey newKey) {
		EdgeElement e = (EdgeElement) newKey;
		this.u = e.u;
		this.v = e.v;
		this.weight = e.weight;
	}

}
