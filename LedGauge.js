// components/LedGauge.js
import React from 'react';
import styles from './LedGauge.module.css';

const LedGauge = ({ score }) => {
  const leds = [];
  const wholePart = Math.floor(score);
  const fractionalPart = score - wholePart;

  for (let i = 1; i <= 10; i++) {
    let color;
    if (i <= wholePart) {
      if (i < 5) {
        color = 'red';
      } else if (i <= 7) {
        color = 'amber';
      } else {
        color = 'green';
      }
    } else if (i === wholePart + 1) {
      if (i <= 4) {
        color = 'red';
      } else if (i <= 7) {
        color = 'amber';
      } else {
        color = 'green';
      }
    } else {
      color = 'gray';
    }

    leds.push(
      <div key={i} className={`${styles.led} ${styles[color]}`}>
        {i <= wholePart ? (
          <div className={styles.fill} style={{ width: '100%' }} />
        ) : i === wholePart + 1 ? (
          <div className={styles.fill} style={{ width: `${fractionalPart * 100}%` }} />
        ) : null}
      </div>
    );
  }

  return <div className={styles.ledContainer}>{leds}</div>;
};

export default LedGauge;
