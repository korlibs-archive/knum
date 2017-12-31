package com.soywiz.knum.opencl

import com.soywiz.knum.KNum
import com.soywiz.knum.KNumContext
import java.nio.Buffer
import java.nio.FloatBuffer

fun <T> KNumOpenCl(forceGpu: Boolean = false, callback: KNum.() -> T) = KNum({ KNumOpenClContext(forceGpu) }, callback)

// @TODO: Maybe we can create custom kernels generating code based on the AST
class KNumOpenClContext(val forceGpu: Boolean = false) : KNumContext() {
    val context: ClContext = ClContext(if (forceGpu) ClContext.DeviceType.FORCE_GPU else ClContext.DeviceType.ANY)
    val queue: ClCommandQueue = context.createCommandQueue()

    override fun close() {
        queue.close()
        context.close()
    }

    override fun <T> computeConstant(tensor: KNum.Constant<T>): KNum.Result<T> {
        return ClBufferResult<T>(tensor.dims, tensor.type, context.createBuffer(tensor.data as FloatBuffer))
    }

    private val programCache = LinkedHashMap<String, ClProgram>()

    private fun generateBinopProgram(op: String, scalar: Boolean) = programCache.getOrPut("myOperation$op$scalar") {
        val l = "a[get_global_id(0)]"
        val r = if (scalar) "b[0]" else "b[get_global_id(0)]"
        context.createProgram("""
            __kernel void myOperation(__global const float *a, __global const float *b, __global float *c) { c[get_global_id(0)] = $l $op $r; }
        """)
    }["myOperation"]

    override fun <T> computeBinaryOp(op: String, l: KNum.Result<T>, r: KNum.Result<T>): KNum.Result<T> {
        val lcl = l as ClBufferResult<T>
        val rcl = r as ClBufferResult<T>
        val kernel = when (op) {
            "add" -> generateBinopProgram("+", rcl.isSingle)
            "sub" -> generateBinopProgram("-", rcl.isSingle)
            "mul" -> generateBinopProgram("*", rcl.isSingle)
            "div" -> generateBinopProgram("/", rcl.isSingle)
            else -> null
        }

        if (kernel != null) {
            //println("Accelerated $op")
            return ClBufferResult(lcl.dims, lcl.type, context.createEmptyBuffer(lcl.type.size, lcl.numElements).apply {
                kernel(queue, lcl.buffer, rcl.buffer, this)
            })
        } else {
            log { "Not accelerated $op" }
            return super.computeBinaryOp(op, l, r)
        }
    }

    private inline fun log(msg: () -> String) {
        println(msg())
    }

    inner class ClBufferResult<T>(dims: IntArray, type: KNum.Type, val buffer: ClBuffer) : KNum.Result<T>(dims, type) {
        override fun getData(): Buffer {
            return FloatBuffer.wrap(buffer.readFloats(queue))
        }
    }
}

object KNumOpenclExample {
    @JvmStatic
    fun main(args: Array<String>) {
        KNumOpenCl {
            val result = floatArrayOf(1f, 2f, 3f, 4f, 5f).const * 10f
            println(result.compute().getFloatArray().toList())
        }
    }
}
