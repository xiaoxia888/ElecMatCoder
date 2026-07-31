import * as XLSX from 'xlsx'
import { downloadBlob, formatPercent } from '@/lib/utils'
import type { EncodingResult, FieldPayload, ImportedRow } from '@/types/encoding'
import { formatFieldCode, formatFieldValue, getDifficultyLevel, getRouteReason, getTypeCategory } from '@/lib/formatters'

const DIFFICULTY_HEADER = '分流最终难度（0=困难，1=中等，2=简单）'

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
    总置信度: isRecognized ? formatPercent(result?.confidence) : '',
    编码正确概率: isRecognized && result?.correctness_confidence?.available
      ? formatPercent(result.correctness_confidence.score)
      : '',
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
    '总置信度',
    '编码正确概率',
    DIFFICULTY_HEADER,
    '分流原因',
    ...EXPORT_FIELDS.flatMap(({ label }) => [`${label}_原始结果`, `${label}_原始编码`]),
  ]
}

export interface ParsedImportPayload {
  rows: Record<string, unknown>[]
  columns: string[]
}

export async function parseExcel(file: File): Promise<ParsedImportPayload> {
  const data = await file.arrayBuffer()
  const workbook = XLSX.read(data)
  const firstSheet = workbook.Sheets[workbook.SheetNames[0]]
  const rows = XLSX.utils.sheet_to_json<Record<string, unknown>>(firstSheet)
  return {
    rows,
    columns: rows[0] ? Object.keys(rows[0]) : [],
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

export function exportStage1Dataset(dataList: ImportedRow[], results: Record<number, EncodingResult>) {
  const rows = dataList
    .map((item) => {
      const result = results[item.index]
      if (!result) return null
      return {
        input: item.text,
        output: Object.fromEntries(
          Object.entries(result.fields || {}).map(([fieldType, field]) => [fieldType, field.stage1_raw?.value ?? null]),
        ),
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
