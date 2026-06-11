import type { BatchJobSummary, ConfigResponse, EncodingResult } from '@/types/encoding'

async function request<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
    ...init,
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `HTTP ${response.status}`)
  }
  return (await response.json()) as T
}

export const api = {
  getConfig() {
    return request<ConfigResponse>('/api/config')
  },
  encodeSingle(payload: { text: string; preprocess?: boolean; project_name?: string }) {
    return request<EncodingResult>('/api/pipe/encode', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },
  createBatchJob(payload: {
    items: Array<{ client_index: number; text: string; project_name?: string; preprocess?: boolean }>
    max_concurrent?: number
  }) {
    return request<{ job: BatchJobSummary }>('/api/pipe/encode/batch/jobs', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },
  listBatchJobs() {
    return request<{ jobs: BatchJobSummary[] }>('/api/pipe/encode/batch/jobs')
  },
  getBatchJob(jobId: string) {
    return request<{ job: BatchJobSummary }>(`/api/pipe/encode/batch/jobs/${jobId}`)
  },
  getBatchJobItem(jobId: string, itemIndex: number) {
    return request<{ result?: EncodingResult }>(`/api/pipe/encode/batch/jobs/${jobId}/items/${itemIndex}`)
  },
  cancelBatchJob(jobId: string) {
    return request<{ job: BatchJobSummary }>(`/api/pipe/encode/batch/jobs/${jobId}/cancel`, {
      method: 'POST',
    })
  },
}
