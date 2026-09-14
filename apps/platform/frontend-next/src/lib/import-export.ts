import * as XLSX from 'xlsx'
import { downloadBlob, formatPercent } from '@/lib/utils'
import type { EncodingResult, FieldPayload, ImportedRow, JsonValue } from '@/types/encoding'
import { formatFieldCode, formatFieldValue, getDifficultyLevel, getRouteReason, getTypeCategory } from '@/lib/formatters'

const DIFFICULTY_HEADER = '分流最终难度（0=困难，2=简单）'

// 导出时占位符 — 视为空值
function blankDash(value: string): string {
  return value === '—' ? '' : value
}

const EXPORT_FIELDS: Array<{ key: string; label: string }> = [
  { key: 'TYPE', label: 'TYPE' },
  { key: 'SIZE', label: 'SIZE' },
  { key: 'THICKNESS', label: 'THICKNESS' },
  { key: 'PRESSURE', label: 'PRESSURE' },
  { key: 'MATERIAL', label: 'MATERIAL' },
  { key: 'STANDARD', label: 'STANDARD' },
]

// 按导出列顺序构造一行数据（CSV / Excel 共用）
function buildExportRecord(item: ImportedRow, result?: EncodingResult): Record<string, string | number> {
  const isRecognized = Boolean(result?.success)
  const difficultyLevel = isRecognized ? getDifficultyLevel(result) : null
  const record: Record<string, string | number> = {
    序号: item.index + 1,
    项目名称: item.projectName || '',
    分类: getTypeCategory(result, item.importedCategory),
    原始描述: item.text,
    原始总编码: isRecognized ? (result?.final_code || '') : '',
    是否需审核: isRecognized ? (result?.need_review ? '是' : '否') : '',
    模型置信分: isRecognized ? formatPercent(result?.confidence) : '',
    [DIFFICULTY_HEADER]: difficultyLevel ?? '',
    分流原因: isRecognized ? getRouteReason(result) : '',
  }
  if (!isRecognized) {
    EXPORT_FIELDS.forEach(({ label }) => {
      record[`${label}_原始结果`] = ''
      record[`${label}_原始编码`] = ''
    })
    return record
  }
  EXPORT_FIELDS.forEach(({ key, label }) => {
    const field = result?.fields?.[key] as FieldPayload | undefined
    // 取二阶段实际输入作为字段值（回退到一阶段原始识别），与最终编码对齐
    const fieldValue = field?.stage2_input?.value ?? field?.stage1_raw?.value
    record[`${label}_原始结果`] = field ? blankDash(formatFieldValue(key, fieldValue)) : ''
    record[`${label}_原始编码`] = field ? blankDash(formatFieldCode(field)) : ''
  })
  return record
}

function getExportRecords(dataList: ImportedRow[], results: Record<number, EncodingResult>) {
  return dataList.map((item) => buildExportRecord(item, results[item.index]))
}

function getExportHeaders(): string[] {
  return [
    '序号',
    '项目名称',
    '分类',
    '原始描述',
    '原始总编码',
    '是否需审核',
    '模型置信分',
    DIFFICULTY_HEADER,
    '分流原因',
    ...EXPORT_FIELDS.flatMap(({ label }) => [`${label}_原始结果`, `${label}_原始编码`]),
  ]
}

export interface ParsedImportPayload {
  rows: Record<string, unknown>[]
  columns: string[]
}

function isBlankImportValue(value: unknown): boolean {
  return value == null || (typeof value === 'string' && value.trim() === '')
}

export function trimTrailingBlankRows(rows: Record<string, unknown>[]): Record<string, unknown>[] {
  let endIndex = rows.length
  while (endIndex > 0 && Object.values(rows[endIndex - 1]).every(isBlankImportValue)) {
    endIndex -= 1
  }
  return endIndex === rows.length ? rows : rows.slice(0, endIndex)
}

export async function parseExcel(file: File): Promise<ParsedImportPayload> {
  const data = await file.arrayBuffer()
  const workbook = XLSX.read(data)
  const firstSheet = workbook.Sheets[workbook.SheetNames[0]]
  // 中间空白行必须保留，只裁掉工作表末尾因格式或历史内容产生的连续空白行。
  const parsedRows = XLSX.utils.sheet_to_json<Record<string, unknown>>(firstSheet, {
    blankrows: true,
    defval: '',
  })
  const rows = trimTrailingBlankRows(parsedRows)
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
  return {
    rows,
    columns,
  }
}

export function exportResultsToCsv(dataList: ImportedRow[], results: Record<number, EncodingResult>) {
  const records = getExportRecords(dataList, results)
  const headers = getExportHeaders()
  const escape = (value: string | number) => `"${String(value).replace(/"/g, '""')}"`
  const lines = [headers.map(escape).join(',')]
  records.forEach((record) => {
    lines.push(headers.map((key) => escape(record[key] ?? '')).join(','))
  })
  downloadBlob('编码结果.csv', new Blob(['\ufeff' + lines.join('\n')], { type: 'text/csv;charset=utf-8;' }))
}

export function exportResultsToExcel(dataList: ImportedRow[], results: Record<number, EncodingResult>) {
  const rows = getExportRecords(dataList, results)
  const headers = getExportHeaders()
  const sheet = XLSX.utils.json_to_sheet(rows, { header: headers })
  const book = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(book, sheet, '编码结果')
  const buffer = XLSX.write(book, { bookType: 'xlsx', type: 'array' })
  downloadBlob('编码结果.xlsx', new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }))
}

function isJsonObject(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}

function isStructuralV2(value: JsonValue | undefined): value is Record<string, JsonValue> {
  return isJsonObject(value) && Array.isArray(value.ITEMS)
}

/**
 * 把页面字段结果还原成一阶段训练源数据。
 *
 * 结构模型的一次 V2 输出会同时挂到 SIZE / THICKNESS / PRESSURE 字段，
 * 供二阶段分别编码。导出时只能保留一份，并恢复成模型训练使用的
 * ITEMS / LENGTH / PRESSURE 顶层结构。
 */
export function buildStage1DatasetOutput(result: EncodingResult): Record<string, JsonValue> {
  const output: Record<string, JsonValue> = {}
  let structural: Record<string, JsonValue> | null = null

  for (const [fieldType, field] of Object.entries(result.fields || {})) {
    const value = field.stage1_raw?.value
    if (['SIZE', 'THICKNESS', 'PRESSURE'].includes(fieldType) && isStructuralV2(value)) {
      structural ??= value
      continue
    }
    output[fieldType] = value ?? null
  }

  if (structural) {
    output.ITEMS = structural.ITEMS ?? []
    output.LENGTH = structural.LENGTH ?? ''
    output.PRESSURE = structural.PRESSURE ?? ''
  }

  return output
}

export function exportStage1Dataset(dataList: ImportedRow[], results: Record<number, EncodingResult>) {
  const rows = dataList
    .map((item) => {
      const result = results[item.index]
      if (!result) return null
      const originalInput = result.original_text || item.text
      // 一阶段推理发生在预处理之后，因此训练用 input 必须与模型实际
      // 收到的 processed_text 一致；original_input 只负责保留格式化前原文。
      const modelInput = result.processed_text || originalInput
      return {
        original_input: originalInput,
        input: modelInput,
        output: buildStage1DatasetOutput(result),
      }
    })
    .filter((item): item is NonNullable<typeof item> => item !== null)
  downloadBlob('一阶段数据集.json', new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json;charset=utf-8;' }))
}

export function resolveProjectName(row: Record<string, unknown>) {
  const normalizedEntries = Object.entries(row).map(([key, value]) => [String(key).trim().toLowerCase().replace(/\s+/g, ''), value] as const)
  const exactKeys = ['项目名称', '子表.项目名称', 'project', 'projectname', '项目', '工程名称', '所属项目', '项目名']
  for (const target of exactKeys.map((item) => item.toLowerCase().replace(/\s+/g, ''))) {
    const hit = normalizedEntries.find(([key]) => key === target)
    if (hit && hit[1] != null && String(hit[1]).trim() !== '') {
      return String(hit[1]).trim()
    }
  }
  return ''
}
