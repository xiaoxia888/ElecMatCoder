import type * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva('inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold', {
  variants: {
    variant: {
      neutral: 'bg-[#efe6d8] text-[#71592d]',
      accent: 'bg-accentSoft text-accent',
      success: 'bg-successSoft text-success',
      caution: 'bg-cautionSoft text-caution',
      danger: 'bg-dangerSoft text-danger',
    },
  },
  defaultVariants: {
    variant: 'neutral',
  },
})

type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
