import { useRef, useState } from 'react'
import { ChevronDown, Upload, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { parseExcel } from '@/lib/import-export'

interface DataImportPanelProps {
  hasData: boolean
  onImport: (rows: Record<string, unknown>[], column: string, fileName: string, projectColumn?: string) => void
}

export function DataImportPanel({ onImport }: DataImportPanelProps) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [rows, setRows] = useState<Record<string, unknown>[]>([])
  const [columns, setColumns] = useState<string[]>([])
  const [selectedColumn, setSelectedColumn] = useState('')
  const [selectedProjectColumn, setSelectedProjectColumn] = useState('')
  const [fileName, setFileName] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')
  const [isDialogOpen, setIsDialogOpen] = useState(false)

  async function processFile(file: File) {
    if (!/\.(xlsx|xls|csv)$/i.test(file.name)) {
      setError('只支持 .csv / .xlsx / .xls 文件')
      return
    }
    setError('')
    setFileName(file.name)
    try {
      const parsed = await parseExcel(file)
      setRows(parsed.rows)
      setColumns(parsed.columns)
      const autoColumn = parsed.columns.find((item) => item.includes('描述')) || parsed.columns[0] || ''
      const autoProjectColumn = parsed.columns.find((item) => /项目|工程|project/i.test(item)) || ''
      setSelectedColumn(autoColumn)
      setSelectedProjectColumn(autoProjectColumn)
      setIsDialogOpen(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : '文件解析失败')
    } finally {
      if (inputRef.current) {
        inputRef.current.value = ''
      }
    }
  }

  function triggerUpload() {
    inputRef.current?.click()
  }

  function handleConfirmImport() {
    if (!selectedColumn || rows.length === 0) return
    onImport(rows, selectedColumn, fileName, selectedProjectColumn)
    setIsDialogOpen(false)
  }

  return (
    <>
      <section className="rounded-xl border border-line bg-white shadow-panel">
        <div className="flex flex-col px-4 pb-4 pt-3.5">
          <h2 className="mb-2.5 text-[15px] font-semibold text-ink">数据导入</h2>

          <button
            type="button"
            onClick={triggerUpload}
            onDragOver={(event) => {
              event.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setIsDragging(false)
              const file = event.dataTransfer.files?.[0]
              if (file) void processFile(file)
            }}
            className={`flex w-full items-center gap-3 rounded-[12px] border border-dashed px-3 py-2.5 text-left transition ${
              isDragging ? 'border-accent bg-accentSoft/20' : 'border-[#c7d2e8] bg-[#fafcff] hover:border-accent'
            }`}
          >
            <Upload className="h-6 w-6 shrink-0 text-accent" />
            <div>
              <div className="text-[13px] text-ink">拖拽文件到此处，或<span className="font-medium text-accent">点击上传</span></div>
              <div className="mt-0.5 text-[12px] leading-4 text-muted">支持 .csv、.xlsx 格式，单次 ≤ 50MB</div>
            </div>
          </button>

          <input
            ref={inputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (file) void processFile(file)
            }}
          />

          {error && <div className="mt-3 rounded-[12px] bg-dangerSoft px-3 py-2 text-sm text-danger">{error}</div>}
        </div>
      </section>

      {isDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/28 px-4">
          <div className="w-full max-w-[420px] rounded-[20px] bg-white shadow-2xl">
            <div className="flex items-center justify-between border-b border-line px-6 py-5">
              <div>
                <h3 className="text-[18px] font-semibold text-ink">确认导入</h3>
                <p className="mt-1 text-sm text-muted">选择材料描述所在列后导入数据。</p>
              </div>
              <button
                type="button"
                className="grid h-8 w-8 place-items-center rounded-md text-muted transition hover:bg-[#f4f6fb] hover:text-ink"
                onClick={() => setIsDialogOpen(false)}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="space-y-4 px-6 py-5">
              <div>
                <div className="mb-2 text-[15px] font-semibold text-ink">描述列</div>
                <div className="relative">
                  <select
                    className="h-12 w-full appearance-none rounded-[14px] border border-line bg-white px-4 pr-10 text-sm text-ink outline-none focus:border-accent"
                    value={selectedColumn}
                    onChange={(event) => setSelectedColumn(event.target.value)}
                  >
                    {columns.length === 0 && <option value="">材料描述</option>}
                    {columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                </div>
              </div>
              <div>
                <div className="mb-2 text-[15px] font-semibold text-ink">项目名称列</div>
                <div className="relative">
                  <select
                    className="h-12 w-full appearance-none rounded-[14px] border border-line bg-white px-4 pr-10 text-sm text-ink outline-none focus:border-accent"
                    value={selectedProjectColumn}
                    onChange={(event) => setSelectedProjectColumn(event.target.value)}
                  >
                    <option value="">不指定（可选）</option>
                    {columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                </div>
              </div>
              <div className="rounded-[14px] bg-[#f7f9fc] px-4 py-3 text-sm text-muted">文件：{fileName}，共 {rows.length} 条数据</div>
              <div className="flex justify-end gap-3">
                <Button variant="outline" className="h-11 rounded-[12px] px-5" onClick={() => setIsDialogOpen(false)}>
                  取消
                </Button>
                <Button variant="accent" className="h-11 rounded-[12px] px-5" disabled={!selectedColumn || rows.length === 0} onClick={handleConfirmImport}>
                  确认导入
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
