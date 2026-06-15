import { Boxes } from 'lucide-react'
import { BatchActionsCard } from '@/components/encoding/BatchActionsCard'
import { CodeResultCard } from '@/components/encoding/CodeResultCard'
import { DataImportPanel } from '@/components/encoding/DataImportPanel'
import { DataListPanel } from '@/components/encoding/DataListPanel'
import { DescriptionCard } from '@/components/encoding/DescriptionCard'
import { ExportCard } from '@/components/encoding/ExportCard'
import { FieldBreakdownCard } from '@/components/encoding/FieldBreakdownCard'
import { RoutingResultCard } from '@/components/encoding/RoutingResultCard'
import { useEncodingWorkspace } from '@/hooks/useEncodingWorkspace'

export function EncodingWorkspace() {
  const workspace = useEncodingWorkspace()

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas text-ink">
      <header className="shrink-0 border-b border-line bg-white">
        <div className="flex h-14 items-center px-6">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-accent text-white shadow-sm">
              <Boxes className="h-5 w-5" />
            </div>
            <div className="text-[18px] font-bold tracking-tight text-[#111827]">材料编码工作台</div>
          </div>
          <nav className="ml-10 flex items-center gap-9">
            <button type="button" className="relative h-14 text-[15px] font-semibold text-accent">
              编码
              <span className="absolute bottom-0 left-0 h-[2.5px] w-full rounded-full bg-accent" />
            </button>
            <button type="button" className="text-[15px] font-semibold text-[#374151]">
              复核
            </button>
          </nav>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-hidden bg-canvas px-4 py-3">
        <div className="grid h-full min-h-0 grid-cols-[300px_minmax(0,1fr)_340px] gap-3">
          {/* 第一列：控制区 */}
          <aside className="flex min-h-0 flex-col gap-3 overflow-hidden">
            <div className="shrink-0">
              <DataImportPanel hasData={workspace.dataList.length > 0} onImport={workspace.importRows} />
            </div>
            <BatchActionsCard
              tasks={workspace.tasks}
              activeTaskId={workspace.activeTaskId}
              onSelectTask={workspace.selectTask}
              canStart={workspace.dataList.length > 0}
              isRunning={workspace.isEncodingBatch}
              isStopping={workspace.isStoppingBatch}
              maxConcurrent={workspace.maxConcurrent}
              defaultMaxConcurrent={workspace.defaultMaxConcurrent}
              onConcurrentChange={workspace.updateMaxConcurrent}
              onStartBatch={workspace.createBatchJob}
              onStopBatch={workspace.cancelBatchJob}
            />
            <ExportCard dataList={workspace.dataList} results={workspace.results} />
          </aside>

          {/* 第二列：详情 */}
          <main className="flex min-h-0 flex-col gap-3 overflow-hidden">
            <div className="shrink-0">
              <DescriptionCard
                currentItem={workspace.currentItem}
                currentResult={workspace.currentResult}
                dataCount={workspace.dataList.length}
                currentIndex={workspace.currentIndex}
                isEncodingSingle={workspace.isEncodingSingle}
                onReencode={workspace.encodeCurrentItem}
                onPrev={workspace.goPrev}
                onNext={workspace.goNext}
                onJump={workspace.goTo}
              />
            </div>
            <div className="shrink-0">
              <CodeResultCard result={workspace.currentResult} />
            </div>
            <div className="min-h-0 flex-1">
              <FieldBreakdownCard result={workspace.currentResult} />
            </div>
            <div className="shrink-0">
              <RoutingResultCard result={workspace.currentResult} />
            </div>
          </main>

          {/* 第三列：数据列表 */}
          <aside className="min-h-0 overflow-hidden">
            <DataListPanel
              dataList={workspace.filteredDataList}
              allItems={workspace.dataList}
              currentIndex={workspace.currentIndex}
              filter={workspace.filter}
              onFilterChange={workspace.setFilter}
              onSelect={workspace.setCurrentIndex}
              getItemStatus={workspace.getItemStatus}
              getItemDifficulty={workspace.getItemDifficulty}
            />
          </aside>
        </div>
      </div>
    </div>
  )
}
