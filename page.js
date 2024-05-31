'use client'
// pages/index.js
import Head from 'next/head';
import LedGauge from '../components/LedGauge';
import { useState } from 'react';

export default function Home() {
  const [score, setScore] = useState(5);

  return (
    <div>
      <Head>
        <title>LED Temperature Gauge</title>
      </Head>

      <main>
        <h1>LED Temperature Gauge</h1>
        <LedGauge score={score} />
        <input
          type="range"
          min="0"
          max="10"
          step="0.1"
          value={score}
          onChange={(e) => setScore(parseFloat(e.target.value))}
        />
      </main>
    </div>
  );
}
