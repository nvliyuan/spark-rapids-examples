/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.spark

import ai.rapids.cudf.{BufferType, ColumnVector, ContiguousTable, HostColumnVector, Table}
import ai.rapids.spark.format.{CodecType, ColumnMeta}
import org.scalatest.FunSuite

import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch

class MetaUtilsSuite extends FunSuite with Arm {
  private def buildContiguousTable(): ContiguousTable = {
    withResource(new Table.TestBuilder()
        .column(5, null.asInstanceOf[java.lang.Integer], 3, 1)
        .column("five", "two", null, null)
        .column(5.0, 2.0, 3.0, 1.0)
        .build()) { table =>
      table.contiguousSplit()(0)
    }
  }

  private def compareColumns(expected: ColumnVector, actual: ColumnVector): Unit = {
    assertResult(expected.getType)(actual.getType)
    assertResult(expected.getRowCount)(actual.getRowCount)
    withResource(expected.copyToHost()) { expectedHost =>
      withResource(actual.copyToHost()) { actualHost =>
        compareColumnBuffers(expectedHost, actualHost, BufferType.DATA)
        compareColumnBuffers(expectedHost, actualHost, BufferType.VALIDITY)
        compareColumnBuffers(expectedHost, actualHost, BufferType.OFFSET)
      }
    }
  }

  private def compareColumnBuffers(
      expected: HostColumnVector,
      actual: HostColumnVector,
      bufferType: BufferType): Unit = {
    val expectedBuffer = expected.getHostBufferFor(bufferType)
    val actualBuffer = actual.getHostBufferFor(bufferType)
    if (expectedBuffer != null) {
      assertResult(expectedBuffer.asByteBuffer())(actualBuffer.asByteBuffer())
    } else {
      assertResult(null)(actualBuffer)
    }
  }

  test("buildTableMeta") {
    withResource(buildContiguousTable()) { contigTable =>
      val table = contigTable.getTable
      val buffer = contigTable.getBuffer
      val meta = MetaUtils.buildTableMeta(7, table, buffer)

      val bufferMeta = meta.bufferMeta
      assertResult(7)(bufferMeta.id)
      assertResult(buffer.getLength)(bufferMeta.compressedSize)
      assertResult(buffer.getLength)(bufferMeta.actualSize)
      assertResult(CodecType.UNCOMPRESSED)(bufferMeta.codec)
      assertResult(table.getRowCount)(meta.rowCount)

      assertResult(table.getNumberOfColumns)(meta.columnMetasLength)
      val columnMeta = new ColumnMeta
      (0 until table.getNumberOfColumns).foreach { i =>
        val col = table.getColumn(i)
        assert(meta.columnMetas(columnMeta, i) != null)
        assertResult(col.getNullCount)(columnMeta.nullCount)
        assertResult(col.getRowCount)(columnMeta.rowCount)
        assertResult(col.getType.getNativeId)(columnMeta.dtype)
        val dataBuffer = col.getDeviceBufferFor(BufferType.DATA)
        assertResult(dataBuffer.getAddress - buffer.getAddress)(columnMeta.data.offset)
        assertResult(dataBuffer.getLength)(columnMeta.data.length)
        val validBuffer = col.getDeviceBufferFor(BufferType.VALIDITY)
        if (validBuffer != null) {
          assertResult(validBuffer.getAddress - buffer.getAddress)(columnMeta.validity.offset)
          assertResult(validBuffer.getLength)(columnMeta.validity.length)
        } else {
          assertResult(null)(columnMeta.validity)
        }
        val offsetsBuffer = col.getDeviceBufferFor(BufferType.OFFSET)
        if (offsetsBuffer != null) {
          assertResult(offsetsBuffer.getAddress - buffer.getAddress)(columnMeta.offsets.offset)
          assertResult(offsetsBuffer.getLength)(columnMeta.offsets.length)
        } else {
          assertResult(null)(columnMeta.offsets)
        }
      }
    }
  }

  test("buildDegenerateTableMeta no columns") {
    val degenerateBatch = new ColumnarBatch(Array(), 127)
    val meta = MetaUtils.buildDegenerateTableMeta(8, degenerateBatch)
    assertResult(null)(meta.bufferMeta)
    assertResult(0)(meta.columnMetasLength)
    assertResult(127)(meta.rowCount)
  }

  test("buildDegenerateTableMeta no rows") {
    val schema = StructType.fromDDL("a INT, b STRING, c DOUBLE")
    withResource(GpuColumnVector.emptyBatch(schema)) { batch =>
      val meta = MetaUtils.buildDegenerateTableMeta(9, batch)
      assertResult(null)(meta.bufferMeta)
      assertResult(0)(meta.rowCount)
      assertResult(3)(meta.columnMetasLength)
      (0 until meta.columnMetasLength).foreach { i =>
        val columnMeta = meta.columnMetas(i)
        assertResult(0)(columnMeta.nullCount)
        assertResult(0)(columnMeta.rowCount)
        val expectedType = batch.column(i).asInstanceOf[GpuColumnVector].getBase.getType
        assertResult(expectedType.getNativeId)(columnMeta.dtype)
        assertResult(null)(columnMeta.data)
        assertResult(null)(columnMeta.validity)
        assertResult(null)(columnMeta.offsets)
      }
    }
  }

  test("getBatchFromMeta") {
    withResource(buildContiguousTable()) { contigTable =>
      val table = contigTable.getTable
      val origBuffer = contigTable.getBuffer
      val meta = MetaUtils.buildTableMeta(10, table, origBuffer)
      withResource(origBuffer.sliceWithCopy(0, origBuffer.getLength)) { buffer =>
        withResource(MetaUtils.getBatchFromMeta(buffer, meta)) { batch =>
          assertResult(table.getRowCount)(batch.numRows)
          assertResult(table.getNumberOfColumns)(batch.numCols)
          (0 until table.getNumberOfColumns).foreach { i =>
            compareColumns(table.getColumn(i),
              batch.column(i).asInstanceOf[GpuColumnVector].getBase)
          }
        }
      }
    }
  }
}
