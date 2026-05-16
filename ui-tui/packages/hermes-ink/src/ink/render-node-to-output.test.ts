import { describe, expect, it } from 'vitest'

import { computeFragmentsForWrappedText } from './render-node-to-output.js'
import type { StyledSegment } from './squash-text-nodes.js'

/**
 * Unit tests for `computeFragmentsForWrappedText` — the helper that
 * emits per-row CachedFragment entries for an ink-text whose segments
 * carry `copySourceFragment` style tags.
 *
 * This is the core of the wrap-aware copy fix: pre-fix, fragments only
 * emitted on row 0, so partial selections across wrap boundaries
 * degraded to block-level mapping. Post-fix, every (segment × row)
 * intersection emits a fragment with the per-row source slice (for
 * verbatim segments) or the whole-segment bounds (for formatted spans).
 */
describe('computeFragmentsForWrappedText', () => {
  const mkSeg = (text: string, tag?: { start: number; end: number; verbatim: boolean }): StyledSegment => ({
    text,
    styles: {} as StyledSegment['styles'],
    ...(tag ? { copySourceFragment: tag } : {})
  })

  it('emits no fragments when no segments carry copySourceFragment', () => {
    const segments = [mkSeg('hello world')]
    const charToSegment = Array(11).fill(0)

    const fragments = computeFragmentsForWrappedText(
      'hello world',
      segments,
      charToSegment,
      'hello world',
      false
    )

    expect(fragments).toEqual([])
  })

  it('verbatim segment: row 0 fragment maps cells 1:1 to source bytes', () => {
    const segments = [mkSeg('hello', { start: 10, end: 15, verbatim: true })]
    const charToSegment = [0, 0, 0, 0, 0]

    const fragments = computeFragmentsForWrappedText(
      'hello',
      segments,
      charToSegment,
      'hello',
      false
    )

    expect(fragments).toEqual([
      { row: 0, colStart: 0, colEnd: 5, start: 10, end: 15, verbatim: true }
    ])
  })

  it('verbatim segment spanning wrap: row 0 + row 1 each get per-row source slice', () => {
    // Single segment "abcdefgh" with source bytes [10, 18) wraps to two
    // rows of width 4. Row 0 = "abcd" → [10, 14). Row 1 = "efgh" → [14, 18).
    const segments = [mkSeg('abcdefgh', { start: 10, end: 18, verbatim: true })]
    const charToSegment = [0, 0, 0, 0, 0, 0, 0, 0]

    const fragments = computeFragmentsForWrappedText(
      'abcd\nefgh',
      segments,
      charToSegment,
      'abcdefgh',
      false
    )

    expect(fragments).toHaveLength(2)
    expect(fragments[0]).toEqual({
      row: 0,
      colStart: 0,
      colEnd: 4,
      start: 10,
      end: 14,
      verbatim: true
    })
    expect(fragments[1]).toEqual({
      row: 1,
      colStart: 0,
      colEnd: 4,
      start: 14,
      end: 18,
      verbatim: true
    })
  })

  it('formatted segment spanning wrap: every row emits whole-segment bounds', () => {
    // Formatted (non-verbatim) segment "XYZWQR" with source bytes [20, 30)
    // wraps across two rows. Both rows should report start=20, end=30 so
    // the copyPointAt snap-rule maps clicks to start or end based on
    // half-width within the on-row part, regardless of which row was hit.
    const segments = [mkSeg('XYZWQR', { start: 20, end: 30, verbatim: false })]
    const charToSegment = [0, 0, 0, 0, 0, 0]

    const fragments = computeFragmentsForWrappedText(
      'XYZ\nWQR',
      segments,
      charToSegment,
      'XYZWQR',
      false
    )

    expect(fragments).toHaveLength(2)
    expect(fragments[0]).toEqual({
      row: 0,
      colStart: 0,
      colEnd: 3,
      start: 20,
      end: 30,
      verbatim: false
    })
    expect(fragments[1]).toEqual({
      row: 1,
      colStart: 0,
      colEnd: 3,
      start: 20,
      end: 30,
      verbatim: false
    })
  })

  it('mixed verbatim + formatted segments wrapping mid-paragraph', () => {
    // verbatim "abcdefgh" source [10, 18) followed by formatted
    // "XYZWQRSTUV" source [20, 30). Wrap at 5 cols:
    //   row 0: "abcde"          → verbatim seg, source [10, 15)
    //   row 1: "fghXY"          → verbatim part [15, 18) + formatted [20, 30)
    //   row 2: "ZWQRS"          → formatted whole-seg [20, 30)
    //   row 3: "TUV"            → formatted whole-seg [20, 30)
    const segments = [
      mkSeg('abcdefgh', { start: 10, end: 18, verbatim: true }),
      mkSeg('XYZWQRSTUV', { start: 20, end: 30, verbatim: false })
    ]

    const charToSegment = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    const fragments = computeFragmentsForWrappedText(
      'abcde\nfghXY\nZWQRS\nTUV',
      segments,
      charToSegment,
      'abcdefghXYZWQRSTUV',
      false
    )

    // Row 0: one verbatim fragment for "abcde".
    const row0 = fragments.filter(f => f.row === 0)
    expect(row0).toHaveLength(1)
    expect(row0[0]).toEqual({ row: 0, colStart: 0, colEnd: 5, start: 10, end: 15, verbatim: true })

    // Row 1: two fragments — verbatim "fgh" then formatted "XY".
    const row1 = fragments.filter(f => f.row === 1).sort((a, b) => a.colStart - b.colStart)
    expect(row1).toHaveLength(2)
    expect(row1[0]).toEqual({ row: 1, colStart: 0, colEnd: 3, start: 15, end: 18, verbatim: true })
    expect(row1[1]).toEqual({ row: 1, colStart: 3, colEnd: 5, start: 20, end: 30, verbatim: false })

    // Row 2 & 3: formatted segment only, whole-segment bounds each row.
    const row2 = fragments.filter(f => f.row === 2)
    expect(row2).toHaveLength(1)
    expect(row2[0]).toEqual({ row: 2, colStart: 0, colEnd: 5, start: 20, end: 30, verbatim: false })

    const row3 = fragments.filter(f => f.row === 3)
    expect(row3).toHaveLength(1)
    expect(row3[0]).toEqual({ row: 3, colStart: 0, colEnd: 3, start: 20, end: 30, verbatim: false })
  })

  it('hard newlines in original advance charIndex correctly across rows', () => {
    // Two-line source separated by \n. Each line is its own visual row
    // — no wrap, but the function still walks them line by line.
    const segments = [mkSeg('hi\nbye', { start: 0, end: 6, verbatim: true })]
    const charToSegment = [0, 0, 0, 0, 0, 0]

    const fragments = computeFragmentsForWrappedText(
      'hi\nbye',
      segments,
      charToSegment,
      'hi\nbye',
      false
    )

    expect(fragments).toHaveLength(2)
    expect(fragments[0]).toEqual({ row: 0, colStart: 0, colEnd: 2, start: 0, end: 2, verbatim: true })
    // After row 0, charIndex skips the '\n' so row 1 starts at byte 3.
    expect(fragments[1]).toEqual({ row: 1, colStart: 0, colEnd: 3, start: 3, end: 6, verbatim: true })
  })
})
