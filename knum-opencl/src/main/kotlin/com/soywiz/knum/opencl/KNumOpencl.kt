package com.soywiz.knum.opencl

import com.soywiz.knum.KNum
import com.soywiz.knum.KNumContext
import java.nio.Buffer
import java.nio.FloatBuffer

fun <T> KNumOpenCl(callback: KNum.() -> T) = KNum({ KNumOpenClContext() }, callback)

class KNumOpenClContext : KNumContext() {
    val context: ClContext = ClContext()
    val queue: ClCommandQueue = context.createCommandQueue()

    override fun close() {
        queue.close()
        context.close()
    }

    inner class BufferResult<T>(dims: IntArray, type: KNum.Type, val buffer: ClBuffer) : KNum.Result<T>(dims, type) {
        override fun getData(): Buffer {
            FloatBuffer.wrap(buffer.readFloats(queue))
            //_data.readFloats()
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
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
