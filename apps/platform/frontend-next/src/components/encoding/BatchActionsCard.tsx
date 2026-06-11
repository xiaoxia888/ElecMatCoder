import { Minus, PauseCircle, PlayCircle, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { TaskInfo } from '@/types/encoding'

interface BatchActionsCardProps {
  tasks: TaskInfo[]
  activeTaskId: string | null
  onSelectTask: (id: string) => void
  canStart: boolean
  isRunning: boolean
  isStopping: boolean
  maxConcurrent: number
  defaultMaxConcurrent: number
  onConcurrentChange: (value: number) => void
  onStartBatch: () => void
  onStopBatch: () => void
}

const STATUS_META: Record<TaskInfo['status'], { label: string; tag: string; bar: string }> = {
  idle: { label: '待处理', tag: 'bg-canvas text-muted', bar: 'bg-[#cbd5e1]' },
  running: { label: '进行中', tag: 'bg-accent text-white', bar: 'bg-accent' },
  partial: { label: '部分完成', tag: 'bg-cautionSoft text-caution', bar: 'bg-caution' },
  done: { label: '已完成', tag: 'bg-successSoft text-success', bar: 'bg-success' },
}

export function BatchActionsCard(props: BatchActionsCardProps) {
  const {
    tasks,
    activeTaskId,
    onSelectTask,
    canStart,
    isRunning,
    isStopping,
    maxConcurrent,
    defaultMaxConcurrent,
    onConcurrentChange,
    onStartBatch,
    onStopBatch,
  } = props

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-xl border border-line bg-white p-4 shadow-panel">
      <div className="mb-2.5 flex shrink-0 items-center justify-between">
        <h2 className="text-[15px] font-bold text-ink">批量操作</h2>
        <span className="text-[12px] text-muted">共 {tasks.length} 个任务</span>
      </div>

      {/* 任务列表（每个任务自带进度），可滚动 */}
      <div className="-mx-1 min-h-0 flex-1 space-y-1.5 overflow-y-auto px-1">
        {tasks.length === 0 ? (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-line text-[13px] text-muted">
            暂无任务，请先导入数据
          </div>
        ) : (
          tasks.map((task) => {
            const meta = STATUS_META[task.status]
            const selected = task.id === activeTaskId
            return (
              <button
                key={task.id}
                type="button"
                onClick={() => onSelectTask(task.id)}
                className={`w-full rounded-xl border p-2.5 text-left transition ${
                  selected ? 'border-accent bg-accentSoft' : 'border-line hover:border-[#c7d2e8]'
                }`}
              >
                <div className="mb-1.5 flex items-center justify-between gap-2">
                  <span className="truncate text-[13px] font-medium text-ink">{task.name}</span>
                  <span className={`shrink-0 rounded-md px-2 py-0.5 text-[12px] font-medium ${meta.tag}`}>{meta.label}</span>
                </div>
                <div className={`mb-1.5 h-1.5 w-full overflow-hidden rounded-full ${selected ? 'bg-white' : 'bg-[#eaeef5]'}`}>
                  <div className={`h-full rounded-full ${meta.bar} transition-all`} style={{ width: `${task.progress}%` }} />
                </div>
                <div className="flex items-center justify-between text-[12px] text-muted">
                  <span>
                    {task.progress}% · {task.total} 条
                  </span>
                  <span>
                    成功 {task.success} · 待审 {task.review}
                  </span>
                </div>
              </button>
            )
          })
        )}
      </div>

      {/* 操作区（固定底部） */}
      <div className="mt-2 shrink-0 border-t border-[#eef1f6] pt-3">
        <Button
          variant={isRunning ? 'danger' : 'accent'}
          size="lg"
          className="h-10 w-full rounded-xl text-[14px] font-semibold"
          onClick={isRunning ? onStopBatch : onStartBatch}
          disabled={isStopping || (!canStart && !isRunning)}
        >
          {isRunning ? <PauseCircle className="mr-2 h-4 w-4" /> : <PlayCircle className="mr-2 h-4 w-4" />}
          {isRunning ? (isStopping ? '停止中…' : '停止编码') : '一键编码当前任务'}
        </Button>

        <div className="mt-3 flex items-center gap-3 text-[13px]">
          <span className="text-muted">并发数</span>
          <div className="ml-1 inline-flex items-center overflow-hidden rounded-lg border border-line">
            <button type="button" className="grid h-8 w-8 place-items-center text-muted transition hover:bg-canvas" onClick={() => onConcurrentChange(maxConcurrent - 1)}>
              <Minus className="h-4 w-4" />
            </button>
            <div className="w-11 text-center text-[13px] font-semibold text-ink">{maxConcurrent}</div>
            <button type="button" className="grid h-8 w-8 place-items-center text-muted transition hover:bg-canvas" onClick={() => onConcurrentChange(maxConcurrent + 1)}>
              <Plus className="h-4 w-4" />
            </button>
          </div>
          <span className="text-[12px] text-muted">默认 {defaultMaxConcurrent}</span>
        </div>
      </div>
    </section>
  )
}
