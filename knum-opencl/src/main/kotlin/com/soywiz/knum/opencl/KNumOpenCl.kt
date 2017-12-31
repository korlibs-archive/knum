package com.soywiz.knum.opencl

import com.soywiz.knum.KNum
import com.soywiz.knum.KNumContext
import java.nio.Buffer

fun <T> KNumOpenCl(callback: KNum.() -> T) = KNum(OpenClKNumContext(), callback)

class OpenClKNumContext : KNumContext() {
    lateinit var context: ClContext

    override fun <T> session(callback: () -> T): T {
        return ClContext().use {
            context = it
            callback()
        }
    }

    inner class BufferResult<T>(dims: IntArray, type: KNum.Type, val _data: ClBuffer) : KNum.Result<T>(dims, type) {
        override fun getData(): Buffer {
            //_data.readFloats()
            TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
        }
    }
}

object KNumOpenClExample {
    @JvmStatic
    fun main(args: Array<String>) {
        KNumOpenCl {
            val result = floatArrayOf(1f, 2f, 3f, 4f, 5f).const * 10f
            println(result.compute().getFloatArray().toList())
        }
    }
}
