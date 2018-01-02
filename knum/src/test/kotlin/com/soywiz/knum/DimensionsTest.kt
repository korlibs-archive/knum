package com.soywiz.knum

import org.junit.Assert.*
import org.junit.Test

class DimensionsTest {
    @Test
    fun index1d() {
        assertEquals(0, Dimensions(2).index(0))
        assertEquals(1, Dimensions(2).index(1))
    }

    @Test
    fun index2d() {
        assertEquals(0, Dimensions(2, 2).index(0, 0))
        assertEquals(1, Dimensions(2, 2).index(0, 1))
        assertEquals(2, Dimensions(2, 2).index(1, 0))
        assertEquals(3, Dimensions(2, 2).index(1, 1))
    }

    @Test
    fun index3d() {
        assertEquals(0, Dimensions(2, 2, 2).index(0, 0, 0))
        assertEquals(1, Dimensions(2, 2, 2).index(0, 0, 1))
        assertEquals(2, Dimensions(2, 2, 2).index(0, 1, 0))
        assertEquals(3, Dimensions(2, 2, 2).index(0, 1, 1))
        assertEquals(4, Dimensions(2, 2, 2).index(1, 0, 0))
        assertEquals(5, Dimensions(2, 2, 2).index(1, 0, 1))
        assertEquals(6, Dimensions(2, 2, 2).index(1, 1, 0))
        assertEquals(7, Dimensions(2, 2, 2).index(1, 1, 1))
    }
}