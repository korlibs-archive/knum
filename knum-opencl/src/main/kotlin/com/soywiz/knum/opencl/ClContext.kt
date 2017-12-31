package com.soywiz.knum.opencl

import org.intellij.lang.annotations.Language
import org.jocl.*
import org.jocl.CL.*
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

class ClContext(val type: DeviceType = DeviceType.ANY) : Closeable {
    enum class DeviceType {
        ANY, FORCE_GPU
    }

    val context: cl_context
    val platformIndex = 0
    val deviceType = when (type) {
        DeviceType.ANY -> CL_DEVICE_TYPE_ALL
        DeviceType.FORCE_GPU -> CL_DEVICE_TYPE_GPU
    }
    val deviceIndex = 0
    val device: cl_device_id

    init {
        setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        device = devices[deviceIndex]!!

        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, arrayOf(device), null, null, null)
    }

    fun createCommandQueue() = ClCommandQueue(this)
    fun createBuffer(data: FloatArray, size: Int = data.size, writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size, writeable)
    fun createBuffer(data: FloatBuffer, size: Int = data.limit(), writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size, writeable)
    fun createBuffer(data: IntBuffer, size: Int = data.limit(), writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_int4, size, writeable)
    fun createEmptyBuffer(elementSize: Int, length: Int, writeable: Boolean = true) = ClBuffer(this, null, elementSize, length, writeable = true)
    fun createProgram(@Language("opencl") source: String) = ClProgram(this, source)

    inline fun <T> queue(callback: ClQueueContext.() -> T): T {
        return createCommandQueue().use { queue ->
            callback(ClQueueContext(queue))
        }
    }

    override fun close() {
        clReleaseContext(context)
    }
}

class ClQueueContext(val queue: ClCommandQueue) {
    fun ClBuffer.readInts() = readInts(queue)
    fun ClBuffer.readFloats() = readFloats(queue)
    //fun ClBuffer.readFloatsQueue() = readFloatsQueue(queue)

    operator fun ClKernel.invoke(vararg args: ClBuffer) = invoke(queue, *args)
    fun ClKernel.invokeQueue(vararg args: ClBuffer) = invokeQueue(queue, *args)
}

class ClBuffer(val ctx: ClContext, val ptr: Pointer?, val elementSize: Int, val length: Int, val writeable: Boolean) : Closeable {
    val sizeInBytes: Int = elementSize * length
    private val flags = run {
        var flags = 0L
        flags = flags or (if (writeable) CL_MEM_READ_WRITE else CL_MEM_READ_ONLY)
        flags = flags or (if (ptr != null) CL_MEM_COPY_HOST_PTR else 0L)
        flags
    }
    val mem = clCreateBuffer(ctx.context, flags, sizeInBytes.toLong(), ptr, null)

    //fun readIntsQueue(queue: ClCommandQueue): IntArray = queue.readByteBuffer(this)
    //fun readFloatsQueue(queue: ClCommandQueue): FloatArray = queue.readByteBuffer(this)

    fun readInts(queue: ClCommandQueue): IntBuffer = queue.readByteBuffer(this).apply { queue.waitCompleted() }.asIntBuffer()
    fun readFloats(queue: ClCommandQueue): FloatBuffer = queue.readByteBuffer(this).apply { queue.waitCompleted() }.asFloatBuffer()

    override fun close() {
        clReleaseMemObject(mem)
    }
}

class ClProgram(val ctx: ClContext, val source: String) : Closeable {
    val result = IntArray(1)
    val program = clCreateProgramWithSource(ctx.context, 1, arrayOf(source), longArrayOf(source.length.toLong()), result)

    init {
        clBuildProgram(program, 0, null, null, null, null)
    }

    operator fun get(name: String) = getKernel(name)
    fun getKernel(name: String) = ClKernel(this, name)

    override fun close() {
        clReleaseProgram(program)
    }
}

class ClKernel(val program: ClProgram, val name: String) {
    operator fun invoke(queue: ClCommandQueue, vararg args: ClBuffer) {
        invokeQueue(queue, *args)
        queue.waitCompleted()
    }

    fun invokeQueue(queue: ClCommandQueue, vararg args: ClBuffer) {
        val errorCode = IntArray(1)
        val kernel = clCreateKernel(program.program, name, errorCode)
        for ((index, arg) in args.withIndex()) {
            clSetKernelArg(kernel, index, Sizeof.cl_mem.toLong(), Pointer.to(arg.mem))
        }
        val global_work_size = longArrayOf(args[0].length.toLong())
        val local_work_size = longArrayOf(1)
        clEnqueueNDRangeKernel(queue.commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null)
        clReleaseKernel(kernel)
    }
}

class ClCommandQueue(val ctx: ClContext) : Closeable {
    val commandQueue = clCreateCommandQueue(ctx.context, ctx.device, 0, null)

    fun readByteBuffer(buffer: ClBuffer): ByteBuffer {
        val out = ByteBuffer.allocate(buffer.sizeInBytes).order(ByteOrder.nativeOrder())
        clEnqueueReadBuffer(commandQueue, buffer.mem, CL_TRUE, 0, buffer.sizeInBytes.toLong(), Pointer.to(out), 0, null, null)
        return out
    }

    fun waitCompleted() {
        clFinish(commandQueue)
    }

    override fun close() {
        waitCompleted()
        clReleaseCommandQueue(commandQueue)
    }
}

object ClExample {
    @JvmStatic
    fun main(args: Array<String>) {
    }
}
