export function createStreamingPcm16Resampler(fromRate, toRate = 16000) {
  const sourceRate = Math.max(1, Number(fromRate) || toRate);
  const targetRate = Math.max(1, Number(toRate) || 16000);
  const ratio = sourceRate / targetRate;
  let sourceOffset = 0;
  let nextOutputPos = 0;
  let previousSample = 0;
  let havePrevious = false;

  const toPcm = (sample) => {
    const value = Math.max(-1, Math.min(1, Number(sample) || 0));
    return value < 0 ? value * 0x8000 : value * 0x7fff;
  };

  return {
    process(input) {
      if (!input?.length) return new Int16Array(0);
      const blockStart = sourceOffset;
      const blockEnd = blockStart + input.length - 1;
      const availableStart = havePrevious ? blockStart - 1 : blockStart;
      const output = [];

      // Keep the fractional conversion phase across audio callbacks. Rounding
      // every 512-sample block independently turns 48kHz into 15937.5Hz, so
      // the 16kHz system channel slowly accumulates behind the user channel.
      while (nextOutputPos <= blockEnd + 1e-9) {
        const local = nextOutputPos - availableStart;
        const i0 = Math.floor(local);
        const frac = local - i0;
        const sampleAt = (index) => {
          if (havePrevious && index === 0) return previousSample;
          const inputIndex = index - (havePrevious ? 1 : 0);
          return input[Math.max(0, Math.min(input.length - 1, inputIndex))];
        };
        const a = sampleAt(i0);
        const b = sampleAt(i0 + 1);
        output.push(toPcm(a * (1 - frac) + b * frac));
        nextOutputPos += ratio;
      }

      sourceOffset += input.length;
      previousSample = input[input.length - 1];
      havePrevious = true;
      return Int16Array.from(output);
    },
  };
}
