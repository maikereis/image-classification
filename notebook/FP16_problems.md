# Principais problemas com loss / gradientes em FP16

1. Underflow (gradientes muito pequenos viram zero)
* O formato FP16 tem uma faixa muito limitada para valores muito pequenos. Segundo a documentação da TensorFlow: valores abaixo de ~ 6 × 10⁻⁸ podem `underflow` para zero.[1]


2. Overflow (gradientes muito grandes viram infinito)
* Se o fator de escala (da loss scaling) for muito grande, durante a retropropagação os gradientes escalados podem exceder o máximo que FP16 representa e virar infinito ou NaN

* O mixed precision (ou `automated mixed precision`) detecta esses NaNs/Inf e pulam a atualização de pesos, além de reduzir o fator de escala dinamicamente, para evitar corromper os pesos.

3. Menor precisão nas atualizações.
* Mesmo quando você escala a loss, a precisão de FP16 é menor (menos bits de fração), então pequenas atualizações de peso podem perder precisão significativa.[3]

4. Loss explode/diverge.
* Se o loss scaling não for bem ajustado, ou se o learning rate + escala forem mal configurados, você pode ver spikes na loss, NaNs, ou divergência no treino.[4]


References:

[1] https://www.tensorflow.org/guide/mixed_precision?utm_source=chatgpt.com

[2] https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html?utm_source=chatgpt.com#scalefactor

[3] https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html?utm_source=chatgpt.com#faq-general__section_on4_qgf_djb

[4] https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html?utm_source=chatgpt.com#mptrain

