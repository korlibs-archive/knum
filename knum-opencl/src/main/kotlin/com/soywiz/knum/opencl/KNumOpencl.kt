package com.soywiz.knum.opencl

import com.soywiz.knum.Dimensions
import com.soywiz.knum.KNum
import com.soywiz.knum.KNumContext
import com.soywiz.knum.opencl.util.ClBuffer
import com.soywiz.knum.opencl.util.ClCommandQueue
import com.soywiz.knum.opencl.util.ClContext
import com.soywiz.knum.opencl.util.ClProgram
import java.nio.Buffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

fun <T> KNumOpenCl(forceGpu: Boolean = false, callback: KNum.() -> T) = KNum({ KNumOpenClContext(forceGpu) }, callback)

// @TODO: Maybe we can create custom kernels generating code based on the AST
class KNumOpenClContext(val forceGpu: Boolean = false) : KNumContext() {
    val context: ClContext = ClContext(if (forceGpu) ClContext.DeviceType.FORCE_GPU else ClContext.DeviceType.ANY)
    val queue: ClCommandQueue = context.createCommandQueue()

    override fun close() {
        queue.close()
        context.close()
    }

    override fun <T> computeConstant(constant: KNum.Constant<T>): KNum.Result<T> {
        val tensorBuffer = constant.data
        return ClBufferResult<T>(constant.dims, constant.type, when (tensorBuffer) {
            is FloatBuffer -> context.createBuffer(tensorBuffer)
            is IntBuffer -> context.createBuffer(tensorBuffer)
            else -> TODO("Unsupported ${constant.data}")
        })
    }

    private val programCache = LinkedHashMap<String, ClProgram>()

    private fun kernelCache(name: String, gen: () -> String) = programCache.getOrPut(name) {
        context.createProgram(gen())
    }["myOperation"]

    private fun binop(op: String, scalar: Boolean) = kernelCache("myOperation$op$scalar") {
        val l = "larray[get_global_id(0)]"
        val r = if (scalar) "rarray[0]" else "rarray[get_global_id(0)]"
        """
            __kernel void myOperation(__global const float *larray, __global const float *rarray, __global float *o) {
                float l = $l;
                float r = $r;
                o[get_global_id(0)] = $op;
            }
        """
    }

    fun <T> KNum.Result<T>.toClBufferResult(): ClBufferResult<T> = when (this) {
        is ClBufferResult<T> -> this
        else -> ClBufferResult<T>(this.dims, this.type, context.createBuffer(this.getFloatArray()))
    }

    override fun <T> computeOperation(op: KNum.Operation<T>): KNum.Result<T> {
        when (op.op) {
            "conv2d" -> {
                val inputResult = compute(op.inputs[0]).toClBufferResult()
                val kernelResult = compute(op.inputs[1]).toClBufferResult()
                val kernel = kernelCache("conv2d") {
                    """
                    #define iindex(x, y) (((y) * iwidth) + (x))
                    #define oindex(x, y) (((y) * owidth) + (x))

                    #define rri(x, y) inp[iindex(x, y)]

                    // @TODO: Vectorize this!
                    __kernel void myOperation(int iwidth, int owidth, __global const float *inp, __global const float *krn, __global float *otp) {
                        int y = get_global_id(0);

                        int outIndex = oindex(0, y);

                        float ma = krn[0], mb = krn[1], mc = krn[2];
                        float md = krn[3], me = krn[4], mf = krn[5];
                        float mg = krn[6], mh = krn[7], mi = krn[8];

                        float a = rri(0, y + 0);
                        float b = rri(1, y + 0);

                        float d = rri(0, y + 1);
                        float e = rri(1, y + 1);

                        float g = rri(0, y + 2);
                        float h = rri(1, y + 2);

                        for (int x = 0; x < owidth; x++) {
                            float c = rri(x + 2, y + 0);
                            float f = rri(x + 2, y + 1);
                            float i = rri(x + 2, y + 2);

                            otp[outIndex + x] =
                                (a * ma) + (b * mb) + (c * mc) +
                                (d * md) + (e * me) + (f * mf) +
                                (g * mg) + (h * mh) + (i * mi)
                            ;

                            a = b; d = e; g = h;
                            b = c; e = f; h = i;
                        }
                    }
                    """
                }
                val output = ClBufferResult<T>(op.dims, op.type, context.createEmptyBuffer(4, op.numElements))
                val istride = inputResult.dims[0]
                val ostride = output.dims[0]

                //println(istride)
                //println(ostride)
                //println(inputResult.getFloatArray().toList())
                //println(kernelResult.getFloatArray().toList())

                kernel.invokeQueue(queue, istride, ostride, inputResult.buffer, kernelResult.buffer, output.buffer, globalWorkRanges = listOf(
                        0L until op.dims[1].toLong()
                ))
                return output
            }
        }
        return super.computeOperation(op)
    }

    override fun <T> computeBinaryOp(op: String, l: KNum.Result<T>, r: KNum.Result<T>): KNum.Result<T> {
        val lcl = l.toClBufferResult()
        val rcl = r.toClBufferResult()
        val kernel = when (op) {
            "add" -> binop("l + r", rcl.isSingle)
            "sub" -> binop("l - r", rcl.isSingle)
            "mul" -> binop("l * r", rcl.isSingle)
            "div" -> binop("l / r", rcl.isSingle)
            "min" -> binop("min(l, r)", rcl.isSingle)
            "max" -> binop("max(l, r)", rcl.isSingle)
            else -> null
        }

        if (kernel != null) {
            //println("Accelerated $op")
            return ClBufferResult(lcl.dims, lcl.type, context.createEmptyBuffer(lcl.type.size, lcl.numElements).apply {
                kernel.invokeQueue(queue, lcl.buffer, rcl.buffer, this)
            })
        } else {
            log { "Not accelerated $op" }
            return super.computeBinaryOp(op, l, r)
        }
    }

    private inline fun log(msg: () -> String) {
        println(msg())
    }

    inner class ClBufferResult<T>(dims: Dimensions, type: KNum.Type, val buffer: ClBuffer) : KNum.Result<T>(dims, type) {
        override fun reshape(dims: Dimensions, type: KNum.Type): ClBufferResult<T> = ClBufferResult(dims, type, buffer)
        override fun getData(): Buffer = when (type) {
            KNum.Type.INT -> buffer.readInts(queue)
            KNum.Type.FLOAT -> buffer.readFloats(queue)
            else -> TODO()
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
