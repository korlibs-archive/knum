package com.soywiz.knum

import org.junit.Test
import kotlin.test.assertEquals

open class KNumTest {
    open protected fun contextGenerate() = KNumContext()

    private fun knumTest(callback: KNum.() -> Unit) = KNum({ contextGenerate() }) {
        callback()
    }

    @Test
    fun addVectors() = knumTest {
        assertEquals(listOf(6f, 8f, 10f, 12f), (floatArrayOf(1f, 2f, 3f, 4f).const + floatArrayOf(5f, 6f, 7f, 8f).const).compute().getFloatArray().toList())
    }

    @Test
    fun addVectorScalar() = knumTest {
        assertEquals(listOf(2f, 3f, 4f, 5f), (floatArrayOf(1f, 2f, 3f, 4f).const + 1f).compute().getFloatArray().toList())
    }

    @Test
    fun mulVectors() = knumTest {
        assertEquals(listOf(-1f, -4f, -9f, -16f), (floatArrayOf(1f, 2f, 3f, 4f).const * floatArrayOf(-1f, -2f, -3f, -4f).const).compute().getFloatArray().toList())
    }

    @Test
    fun mulVectorScalar() = knumTest {
        assertEquals(listOf(2f, 4f, 6f, 8f), (floatArrayOf(1f, 2f, 3f, 4f).const * 2f).compute().getFloatArray().toList())
    }

    @Test
    fun minVectors() = knumTest {
        assertEquals(listOf(-1f, -1f, -1f, 1f, -2f, 2f, -1f, 0f), min(floatArrayOf(-1f, 1f, -1f, 1f, 2f, 2f, 0f, 0f).const, floatArrayOf(1f, -1f, -1f, 1f, -2f, 2f, -1f, 0f).const).compute().getFloatArray().toList())
    }

    @Test
    fun maxVectors() = knumTest {
        assertEquals(listOf(1f, 1f, -1f, 1f, 2f, 2f, 0f, 0f), max(floatArrayOf(-1f, 1f, -1f, 1f, 2f, 2f, 0f, 0f).const, floatArrayOf(1f, -1f, -1f, 1f, -2f, 2f, -1f, 0f).const).compute().getFloatArray().toList())
    }

    @Test
    fun pad() = knumTest {
        assertEquals(
                listOf(
                        0f, 0f, 0f, 0f,
                        0f, 1f, 2f, 0f,
                        0f, 3f, 4f, 0f,
                        0f, 0f, 0f, 0f
                ),
                floatArrayOf(1f, 2f, 3f, 4f).const.reshape(2, 2).pad(1, 1).compute().getFloatArray().toList()
        )
    }

    @Test
    open fun conv2d() = knumTest {
        assertEquals(
                listOf(
                        79f, 67f,
                        51f, 37f
                ),
                floatArrayOf(1f, 2f, 3f, 4f).const.reshape(2, 2).pad(1, 1).conv2d(floatArrayOf(
                        1f, 2f, 3f,
                        4f, 5f, 6f,
                        7f, 8f, 9f
                ).const(3, 3)).compute().getFloatArray().toList()
        )
    }

    @Test
    open fun conv2db() = knumTest {
        assertEquals(
                listOf(
                        1f, 2f,
                        3f, 4f
                ),
                floatArrayOf(1f, 2f, 3f, 4f).const.reshape(2, 2).pad(1, 1).conv2d(floatArrayOf(
                        0f, 0f, 0f,
                        0f, 1f, 0f,
                        0f, 0f, 0f
                ).const(3, 3)).compute().getFloatArray().toList()
        )
    }

    @Test
    open fun conv2dc() = knumTest {
        assertEquals(
                listOf(
                        4f, 0f,
                        0f, 0f
                ),
                floatArrayOf(1f, 2f, 3f, 4f).const.reshape(2, 2).pad(1, 1).conv2d(floatArrayOf(
                        0f, 0f, 0f,
                        0f, 0f, 0f,
                        0f, 0f, 1f
                ).const(3, 3)).compute().getFloatArray().toList()
        )
    }
}