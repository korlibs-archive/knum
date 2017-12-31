package com.soywiz.knum.opencl

import org.junit.Test
import kotlin.test.assertEquals

class ClContextTest {
    @Test
    fun name() {
        val result = ClContext().run {
            queue {
                val program = createProgram("""
                    __kernel void sampleKernel(__global const float *a, __global const float *b, __global float *c) {
                        int gid = get_global_id(0);
                        c[gid] = a[gid] * b[gid] * b[gid] * 2.0;
                    }
                """)

                createBuffer(floatArrayOf(1f, 2f, 3f, 4f)).use { buffer1 ->
                    createBuffer(floatArrayOf(5f, 6f, 7f, 8f)).use { buffer2 ->
                        createEmptyBuffer(4, 4, writeable = true).use { buffer3 ->
                            program["sampleKernel"](buffer1, buffer2, buffer3)
                            buffer3.readFloats()
                        }
                    }
                }
            }
        }

        assertEquals(listOf(50f, 144f, 294f, 512f), result.toList())
    }
}