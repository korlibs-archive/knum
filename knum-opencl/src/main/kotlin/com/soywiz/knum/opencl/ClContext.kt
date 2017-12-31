package com.soywiz.knum.opencl

import org.jocl.*
import java.io.Closeable


class ClContext : Closeable {
    val context: cl_context
    val platformIndex = 0
    val deviceType = CL.CL_DEVICE_TYPE_ALL
    val deviceIndex = 0
    val device: cl_device_id

    init {
        CL.setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        CL.clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        CL.clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        device = devices[deviceIndex]!!

        // Create a context for the selected device
        context = CL.clCreateContext(contextProperties, 1, arrayOf<cl_device_id>(device), null, null, null)
    }

    fun createCommandQueue() = ClCommandQueue(this)
    fun createBuffer(data: FloatArray, size: Int = data.size) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size)

    override fun close() {
        CL.clReleaseContext(context)
    }
}

class ClBuffer(val ctx: ClContext, val ptr: Pointer, val elementSize: Int, val length: Int) : Closeable {
    val sizeInBytes: Int = elementSize * length
    val mem = CL.clCreateBuffer(ctx.context, CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR, sizeInBytes.toLong(), ptr, null)

    fun readFloats(queue: ClCommandQueue): FloatArray = queue.readFloats(this)

    override fun close() {
        CL.clReleaseMemObject(mem)
    }
}

class ClProgram(val ctx: ClContext, val source: String) : Closeable {
    val result = IntArray(1)
    val program = CL.clCreateProgramWithSource(ctx.context, 1, arrayOf(source), longArrayOf(source.length.toLong()), result)

    fun getKernel(name: String) = ClKernel(this, name)

    override fun close() {
        CL.clReleaseProgram(program)
    }
}

class ClKernel(val program: ClProgram, val name: String) : Closeable {
    val errorCode = IntArray(1)
    val kernel = CL.clCreateKernel(program.program, name, errorCode)

    override fun close() {
        CL.clReleaseKernel(kernel)
    }
}

class ClCommandQueue(val ctx: ClContext) : Closeable {
    val commandQueue = CL.clCreateCommandQueue(ctx.context, ctx.device, 0, null)

    fun readFloats(buffer: ClBuffer): FloatArray {
        val out = FloatArray(buffer.length)
        CL.clEnqueueReadBuffer(commandQueue, buffer.mem, CL.CL_TRUE, 0, buffer.sizeInBytes.toLong(), Pointer.to(out), 0, null, null)
        return out
    }

    override fun close() {
        CL.clReleaseCommandQueue(commandQueue)
    }
}
