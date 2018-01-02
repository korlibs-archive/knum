package com.soywiz.knum

import java.io.Closeable
import java.nio.Buffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

object KNumExample {
    @JvmStatic
    fun main(args: Array<String>) {
        val res = KNum {
            //println(floatArrayOf(1f, 2f, 3f, 4f).const)
            //println(floatArrayOf(1f, 2f, 3f, 4f).const.reshape(2, 2))
            val tensor = floatArrayOf(1f, 2f, 3f, 4f).const + floatArrayOf(4f, 5f, 6f, 7f).const
            val result = (tensor * -(1f)).clamp(-8f, -6f).compute().getFloatArray()
            println(result.toList())
            10
        }
    }
}

// @TODO: Convert AST into other IR so we can clean old stuff to use less memory
open class KNumContext : Closeable {
    class DefaultResult<T>(dims: Dimensions, type: KNum.Type, val _data: Buffer) : KNum.Result<T>(dims, type) {
        override fun getData(): Buffer = _data
        override fun reshape(dims: Dimensions, type: KNum.Type): KNum.Result<T> = DefaultResult(dims, type, _data)
    }

    override fun close() {
    }

    open fun <T> computeRoot(tensor: KNum.Tensor<T>): KNum.Result<T> {
        return compute(tensor)
    }

    protected fun <T> compute(tensor: KNum.Tensor<T>): KNum.Result<T> {
        return when (tensor) {
            is KNum.Constant -> computeConstant(tensor)
            is KNum.Operation -> computeOperation(tensor)
            else -> TODO("Don't know how to compute $tensor")
        }
    }

    open protected fun <T> computeConstant(constant: KNum.Constant<T>): KNum.Result<T> {
        return DefaultResult<T>(constant.dims, constant.type, constant.data)
    }

    open protected fun <T> computeOperation(op: KNum.Operation<T>): KNum.Result<T> = op.run {
        when (this.op) {
            "reshape" -> compute(inputs[0] as KNum.Tensor<T>).reshape(op.dims, op.type)
            "add", "sub", "mul", "div", "min", "max" -> computeBinaryOp<T>(this.op, compute(inputs[0] as KNum.Tensor<T>), compute(inputs[1] as KNum.Tensor<T>))
            "neg" -> computeUnaryOp<T>(this.op, compute(inputs[0] as KNum.Tensor<T>))
            "pad" -> {
                val itensor = compute(inputs[0] as KNum.Tensor<T>)
                val pad = compute(inputs[1] as KNum.Tensor<Int>).getIntArray()
                when (itensor.rank) {
                    2 -> {
                        val i = Float2Transfer(itensor.dims, itensor.getFloatBuffer())
                        val o = Float2Transfer(Dimensions(itensor.dims[0] + pad[0] * 2, itensor.dims[1] + pad[1] * 2))
                        val out = DefaultResult<T>(o.dims, itensor.type, o.buffer)
                        val padX = pad[0]
                        val padY = pad[1]
                        for (y in 0 until i.height) {
                            for (x in 0 until i.width) {
                                o[x + padX, y + padY] = i[x, y]
                            }
                        }
                        out
                    }
                    else -> TODO("Just supported tensors of rank 2")
                }
            }
            "conv2d" -> {
                val itensor = compute(inputs[0] as KNum.Tensor<Float>)
                val kernel = compute(inputs[1] as KNum.Tensor<Float>)
                if (itensor.dims.rank != 2) throw IllegalArgumentException("Just supporting tensors of rank 2 right now")
                if (!kernel.dims.match(3, 3)) throw IllegalArgumentException("Just supported kernels of 3x3")
                val inp = Float2Transfer(itensor.dims, itensor.getFloatBuffer())
                val out = Float2Transfer(Dimensions(inp.width - 2, inp.height - 2))
                val krn = Float2Transfer(kernel.dims, kernel.getFloatBuffer())

                val am = krn[0, 0]
                val bm = krn[1, 0]
                val cm = krn[2, 0]

                val dm = krn[0, 1]
                val em = krn[1, 1]
                val fm = krn[2, 1]

                val gm = krn[0, 2]
                val hm = krn[1, 2]
                val im = krn[2, 2]


                for (y in 0 until out.height) {
                    var a = inp[0, y]
                    var b = inp[1, y]

                    var d = inp[0, y + 1]
                    var e = inp[1, y + 1]

                    var g = inp[0, y + 2]
                    var h = inp[1, y + 2]

                    for (x in 0 until out.width) {
                        val c = inp[x + 2, y]
                        val f = inp[x + 2, y + 1]
                        val i = inp[x + 2, y + 2]

                        out[x, y] = (a * am) + (b * bm) + (c * cm) + (d * dm) + (e * em) + (f * fm) + (g * gm) + (h * hm) + (i * im)

                        a = b
                        d = e
                        g = h

                        b = c
                        e = f
                        h = i
                    }
                }
                DefaultResult(dims, KNum.Type.FLOAT, out.buffer)
            }
            else -> TODO("Unsuported operation ${this.op}")
        }
    }

    open protected fun <T> computeUnaryOp(op: String, l: KNum.Result<T>): KNum.Result<T> {
        val lf = l.getData() as FloatBuffer
        val num = l.numElements
        val fop = when (op) {
            "neg" -> ::fneg
            else -> TODO("Unsupported operation $op")
        }
        return DefaultResult<T>(l.dims, l.type, FloatBuffer.allocate(l.numElements).apply {
            for (n in 0 until num) put(n, fop(lf[n]))
        })
    }

    class Float2Transfer(val dims: Dimensions, val buffer: FloatBuffer = FloatBuffer.allocate(dims.numElements)) {
        val width = dims[0]
        val height = dims[1]
        fun index(x: Int, y: Int) = x + (y * width)
        operator fun get(x: Int, y: Int) = buffer[index(x, y)]
        operator fun set(x: Int, y: Int, value: Float) = run { buffer.put(index(x, y), value) }
    }

    class Float3Transfer(val buffer: FloatBuffer, val width: Int, val height: Int, val depth: Int) {
        fun index(x: Int, y: Int, z: Int) = x + ((y + (z * height)) * width)
        operator fun get(x: Int, y: Int, z: Int) = buffer[index(x, y, z)]
        operator fun set(x: Int, y: Int, z: Int, value: Float) = run { buffer.put(index(x, y, z), value) }
    }

    protected fun fneg(l: Float): Float = -l
    protected fun fadd(l: Float, r: Float): Float = l + r
    protected fun fsub(l: Float, r: Float): Float = l - r
    protected fun fmul(l: Float, r: Float): Float = l * r
    protected fun fdiv(l: Float, r: Float): Float = l / r
    protected fun fmax(l: Float, r: Float): Float = kotlin.math.max(l, r)
    protected fun fmin(l: Float, r: Float): Float = kotlin.math.min(l, r)

    open protected fun <T> computeBinaryOp(op: String, l: KNum.Result<T>, r: KNum.Result<T>): KNum.Result<T> {
        val leftBuffer = l.getData() as FloatBuffer
        val rightBuffer = r.getData() as FloatBuffer
        val num = l.numElements

        fun getL_multi(n: Int) = leftBuffer[n]
        fun getR_single(n: Int) = rightBuffer[0]
        fun getR_multi(n: Int) = rightBuffer[n]

        val getL = ::getL_multi
        val getR = if (r.isSingle) ::getR_single else ::getR_multi

        val fop = when (op) {
            "add" -> ::fadd
            "sub" -> ::fsub
            "mul" -> ::fmul
            "div" -> ::fdiv
            "max" -> ::fmax
            "min" -> ::fmin
            else -> TODO("Unsuported operation $op")
        }

        return DefaultResult(l.dims, l.type, FloatBuffer.allocate(l.numElements).apply {
            for (n in 0 until num) put(n, fop(getL(n), getR(n)))
        })
    }
}

class Dimensions(vararg val values: Int) {
    constructor(values: Iterable<Int>) : this(*values.toList().toIntArray())
    fun map(callback: (value: Int) -> Int): Dimensions = Dimensions(values.map(callback))
    val rank: Int get() = values.size
    val numElements: Int by lazy { values.reduce { acc, i -> acc * i } }
    val isSingle: Boolean get() = values.size == 1 && values[0] == 1
    fun match(vararg dims: Int) = (rank == dims.size) && (0 until dims.size).all { dims[it] == this[it] }
    fun toList() = values.toList()
    operator fun get(index: Int) = values[index]
    override fun toString(): String = values.joinToString(", ")
}

fun IntArray.toDimensions() = Dimensions(*this)
fun List<Int>.toDimensions() = Dimensions(*this.toIntArray())

class KNum(val ctx: KNumContext) {
    enum class Type(val size: Int) { INT(4), FLOAT(4) }

    abstract class Tensor<T>(val dims: Dimensions, val type: Type) {
        val rank: Int get() = dims.rank
        val numElements: Int get() = dims.numElements
        val isSingle: Boolean get() = dims.isSingle
        override fun toString(): String = "Tensor[$type]($dims)"
    }

    abstract class Result<T>(dims: Dimensions, type: Type) : Tensor<T>(dims, type) {
        abstract fun reshape(dims: Dimensions, type: Type): Result<T>
        abstract fun getData(): Buffer

        fun getFloatBuffer(): FloatBuffer = getData() as FloatBuffer
        fun getIntBuffer(): IntBuffer = getData() as IntBuffer
        fun getFloatArray(): FloatArray = getFloatBuffer().run { FloatArray(limit()).apply { position(0); get(this) } }
        fun getIntArray(): IntArray = getIntBuffer().run { IntArray(limit()).apply { position(0); get(this) } }
    }

    class Operation<T>(val op: String, type: Type, dims: Dimensions, val inputs: Array<Tensor<*>>) : Tensor<T>(dims, type) {
        override fun toString(): String = "Operation($op[$type], ${dims.toList()})(${inputs.toList()})"
    }

    class Constant<T>(dims: Dimensions, type: Type, val data: Buffer) : Tensor<T>(dims, type) {
        init {
            if (numElements != data.limit()) {
                throw IllegalArgumentException("${dims.toList()}")
            }
        }
    }

    val IntArray.const: Constant<Int> get() = Constant(Dimensions(this.size), Type.INT, IntBuffer.wrap(this))
    val FloatArray.const: Constant<Float> get() = Constant(Dimensions(this.size), Type.FLOAT, FloatBuffer.wrap(this))
    fun FloatArray.const(dims: Dimensions) = const.reshape(dims)
    fun FloatArray.const(vararg dims: Int) = const.reshape(*dims)

    val Int.const: Constant<Int> get() = Constant(Dimensions(1), Type.INT, IntBuffer.wrap(intArrayOf(this)))
    val Float.const: Constant<Float> get() = Constant(Dimensions(1), Type.FLOAT, FloatBuffer.wrap(floatArrayOf(this)))

    val <T> T.const: Constant<T>
        get() = when (this) {
            is IntArray -> this.const as Constant<T>
            is FloatArray -> this.const as Constant<T>
            is Int -> this.const as Constant<T>
            is Float -> this.const as Constant<T>
            else -> throw IllegalArgumentException("Unsupported $this")
        }

    fun <T> max(l: Tensor<T>, r: Tensor<T>): Tensor<T> = Operation<T>("max", l.type, l.dims, arrayOf(l, r))
    fun <T> min(l: Tensor<T>, r: Tensor<T>): Tensor<T> = Operation<T>("min", l.type, l.dims, arrayOf(l, r))

    fun <T> Tensor<T>.clamp(min: Tensor<T>, max: Tensor<T>): Tensor<T> = min(max(this, min), max)
    fun <T> Tensor<T>.clamp(min: T, max: T): Tensor<T> = min(max(this, min.const), max.const)

    operator fun <T> Tensor<T>.unaryMinus(): Tensor<T> = Operation<T>("neg", this.type, this.dims, arrayOf(this))

    operator fun <T> Tensor<T>.times(that: Tensor<T>): Tensor<T> = Operation<T>("mul", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.div(that: Tensor<T>): Tensor<T> = Operation<T>("div", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.plus(that: Tensor<T>): Tensor<T> = Operation<T>("add", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.minus(that: Tensor<T>): Tensor<T> = Operation<T>("sub", this.type, this.dims, arrayOf(this, that))
    operator fun <T> Tensor<T>.times(that: T): Tensor<T> = Operation<T>("mul", this.type, this.dims, arrayOf(this, that.const))
    operator fun <T> Tensor<T>.div(that: T): Tensor<T> = Operation<T>("div", this.type, this.dims, arrayOf(this, that.const))
    operator fun <T> Tensor<T>.plus(that: T): Tensor<T> = Operation<T>("add", this.type, this.dims, arrayOf(this, that.const))
    operator fun <T> Tensor<T>.minus(that: T): Tensor<T> = Operation<T>("sub", this.type, this.dims, arrayOf(this, that.const))

    fun <T> Tensor<T>.reshape(dims: Dimensions): Tensor<T> = Operation<T>("reshape", this.type, dims, arrayOf(this))
    fun <T> Tensor<T>.reshape(vararg dims: Int): Tensor<T> = reshape(dims.toDimensions())
    fun <T> Tensor<T>.transpose(vararg axis: Int): Tensor<T> = Operation<T>("transpose", this.type, Dimensions(axis.map { this.dims.values[it] }), arrayOf(this))
    fun <T> Tensor<T>.pad(vararg pads: Int): Tensor<T> = Operation<T>("pad", this.type, Dimensions((0 until pads.size).map { dims.values[it] + pads[it] * 2 }), arrayOf(this, pads.const))

    fun <T> Tensor<T>.conv2d(kernel: Tensor<T>): Tensor<T> = Operation<T>("conv2d", this.type, dims.map { it - 2 }, arrayOf(this, kernel))

    fun <T> Tensor<T>.compute(): Result<T> = ctx.computeRoot(this)
}

fun <T> KNum(contextGenerator: () -> KNumContext = { KNumContext() }, callback: KNum.() -> T): T {
    return contextGenerator().use { context ->
        callback(KNum(context))
    }
}
