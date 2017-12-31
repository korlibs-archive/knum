package com.soywiz.knum

import org.junit.Test
import kotlin.test.assertEquals

open class KNumTest {
    open protected fun contextGenerate() = KNumContext()

    @Test
    fun vectorsSum() = KNum({ contextGenerate() }) {
        assertEquals(listOf(6f, 8f, 10f, 12f), (floatArrayOf(1f, 2f, 3f, 4f).const + floatArrayOf(5f, 6f, 7f, 8f).const).compute().getFloatArray().toList())
    }

    @Test
    fun vectorBias() = KNum({ contextGenerate() }) {
        assertEquals(listOf(2f, 3f, 4f, 5f), (floatArrayOf(1f, 2f, 3f, 4f).const + 1f).compute().getFloatArray().toList())
    }
}