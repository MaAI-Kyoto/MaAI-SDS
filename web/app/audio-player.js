/** Streaming PCM16 (16 kHz mono LE) playback via the Web Audio API. */

export function createStreamingFloatResampler(fromRate, toRate) {
  const sourceRate = Math.max(1, Number(fromRate) || 16000);
  const targetRate = Math.max(1, Number(toRate) || sourceRate);
  const ratio = sourceRate / targetRate;
  let sourceOffset = 0;
  let nextOutputPos = 0;
  let previousSample = 0;
  let havePrevious = false;

  return {
    process(input) {
      if (!input?.length) return new Float32Array(0);
      if (sourceRate === targetRate) return input.slice();
      const blockStart = sourceOffset;
      const blockEnd = blockStart + input.length - 1;
      const availableStart = havePrevious ? blockStart - 1 : blockStart;
      const output = [];
      const sampleAt = (index) => {
        if (havePrevious && index === 0) return previousSample;
        const inputIndex = index - (havePrevious ? 1 : 0);
        return input[Math.max(0, Math.min(input.length - 1, inputIndex))];
      };
      while (nextOutputPos <= blockEnd + 1e-9) {
        const local = nextOutputPos - availableStart;
        const i0 = Math.floor(local);
        const frac = local - i0;
        const a = sampleAt(i0);
        const b = sampleAt(i0 + 1);
        output.push(a * (1 - frac) + b * frac);
        nextOutputPos += ratio;
      }
      sourceOffset += input.length;
      previousSample = input[input.length - 1];
      havePrevious = true;
      return Float32Array.from(output);
    },
  };
}

export function createAudioPlayer() {
  /** @type {AudioContext | null} */
  let ctx = null;
  let nextStartTime = 0;
  let outputResampler = null;
  let outputSampleRate = 0;
  const SAMPLE_RATE = 16000;
  // Windows の WAV 再生と同じだけ先読みし、20 ms フレームの配送ジッタによる
  // AudioContext のアンダーランを防ぐ。
  const REBUFFER_LEAD_SEC = 0.08;

  async function ensureContext() {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      // 出力デバイスの既定レートで作り、連続状態を保ったリサンプラーを手前に置く。
      ctx = new AC();
    }
    if (ctx.state === 'suspended') await ctx.resume();
    return ctx;
  }

  function pcm16ToFloat32(pcm) {
    let i16;
    if (pcm instanceof Int16Array) {
      i16 = pcm;
    } else if (pcm instanceof ArrayBuffer) {
      i16 = new Int16Array(pcm);
    } else if (ArrayBuffer.isView(pcm)) {
      i16 = new Int16Array(pcm.buffer, pcm.byteOffset, Math.floor(pcm.byteLength / 2));
    } else {
      throw new TypeError('Expected Int16Array or ArrayBuffer');
    }
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) {
      f32[i] = Math.max(-1, Math.min(1, i16[i] / 32768));
    }
    return f32;
  }

  function enqueuePcm16(pcm) {
    if (!ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      ctx = new AC();
    }
    const sourceData = pcm16ToFloat32(pcm);
    if (sourceData.length === 0) return null;

    if (!outputResampler || outputSampleRate !== ctx.sampleRate) {
      outputSampleRate = ctx.sampleRate;
      outputResampler = createStreamingFloatResampler(SAMPLE_RATE, outputSampleRate);
    }
    // AudioBufferSource をフレームごとに作ってもクリック音が出ないよう、先に
    // 連続状態のままデバイスレートへ変換する。
    const floatData = outputResampler.process(sourceData);
    if (floatData.length === 0) return null;

    const play = () => {
      if (!ctx || ctx.state === 'closed' || ctx.state === 'suspended') return null;
      const buffer = ctx.createBuffer(1, floatData.length, outputSampleRate);
      buffer.copyToChannel(floatData, 0);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const now = ctx.currentTime;
      if (nextStartTime < now) nextStartTime = now + REBUFFER_LEAD_SEC;
      source.start(nextStartTime);
      nextStartTime += buffer.duration;
      return true;
    };

    if (ctx.state === 'suspended') {
      ctx.resume().then(play).catch(() => {});
      return null;
    }
    return play();
  }

  function reset() {
    outputResampler = null;
    outputSampleRate = 0;
    nextStartTime = ctx ? ctx.currentTime : 0;
  }

  async function close() {
    if (!ctx) return;
    try {
      await ctx.close();
    } catch (_) {
      /* ignore */
    }
    ctx = null;
    nextStartTime = 0;
    outputResampler = null;
    outputSampleRate = 0;
  }

  return {
    ensureContext,
    enqueuePcm16,
    reset,
    close,
    getContext: () => ctx,
  };
}
