export type Primitive = string | number | boolean | null
export type JsonValue = Primitive | JsonValue[] | { [key: string]: JsonValue }

export interface ImportedRow {
  index: number
  text: string
  projectName: string
  rawRow: Record<string, unknown>
}

export interface Stage1RawPayload {
  value: JsonValue
  source?: string
  confidence?: number | null
  reason?: string
  evidence?: Record<string, unknown>
}

export interface Stage2InputPayload {
  value: JsonValue
  notes?: string[]
}

export interface Stage2OutputPayload {
  code: string
}

export interface ConfidenceDetail {
  stage1?: number | null
  stage2?: number | null
  field?: number | null
}

export interface FieldStatus {
  need_review?: boolean
  similarity?: number | null
  is_exact_match?: boolean | null
}

export interface FieldPayload {
  field_type: string
  stage1_raw: Stage1RawPayload
  stage2_input: Stage2InputPayload
  stage2_output: Stage2OutputPayload
  confidence_detail?: ConfidenceDetail
  status?: FieldStatus
}

export interface EncodingResult {
  original_text: string
  processed_text?: string
  final_code: string
  success: boolean
  need_review: boolean
  confidence?: number
  fields: Record<string, FieldPayload>
  route_info?: Record<string, unknown> | null
  routing?: {
    stage1_level?: number | null
    final_level?: number | null
    decision_stage?: 'stage1' | 'second_pass' | 'project_frequency' | string
    need_review?: boolean
    reason_text?: string
    failed_checks?: Array<{
      field?: string
      reason?: string
      stage?: string
      rule?: string
    }>
    passed_checks?: string[]
    missing_required_checks?: string[]
  } | null
  difficulty_split?: {
    level?: number | null
    difficulty?: number | null
    reason_text?: string
    reasons?: string[]
    failed_checks?: Array<{
      field?: string
      reason?: string
      stage?: string
      rule?: string
    }>
    passed_checks?: string[]
  } | null
  second_pass?: {
    stage1_level?: number | null
    stage1_difficulty?: number | null
    final_level?: number | null
    decision_stage?: string
    need_review?: boolean
    reason_text?: string
    failed_checks?: Array<{
      field?: string
      reason?: string
      stage?: string
      rule?: string
    }>
    passed_checks?: string[]
    missing_required_checks?: string[]
  } | null
  errors?: string[]
  warnings?: string[]
}

export interface BatchJobSummary {
  job_id: string
  status: string
  total: number
  processed: number
  success_count?: number
  review_count?: number
  max_concurrent?: number
  queue_position?: number | null
  created_at?: number
  finished_at?: number | null
  items?: Array<{ index?: number; text?: string; project_name?: string }>
  results?: Record<string, EncodingResult>
}

export interface BatchJobEvent {
  type: 'snapshot' | 'item' | 'end' | 'cancelled' | 'failed'
  index?: number
  result?: EncodingResult
  snapshot?: BatchJobSummary
  error?: string
}

export interface ConfigResponse {
  batch_processing?: {
    max_concurrent?: number
  }
}

export interface TaskInfo {
  id: string
  name: string
  total: number
  success: number
  review: number
  progress: number
  status: 'idle' | 'running' | 'partial' | 'done'
}
