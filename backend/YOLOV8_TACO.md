# YOLOv8/TACO em um caminho independente

O novo comando usa **YOLOv8s treinado em resíduos TACO**, exportado em ONNX.
Ele não é SSD MobileNet e não substitui o detector YOLOv26 existente.

## Como a integração foi feita

A análise encontrou um monitor comum para os detectores existentes,
`backend.monitoring.monitor_residuos`, com adaptadores em `backend/detection`.
Esse monitor, seus adaptadores e suas configurações foram preservados.

O caminho novo tem seus próprios arquivos:

- Adaptador: `backend/detection/yolov8_taco_detector.py`.
- Monitor: `backend/monitoring/monitor_yolov8_taco.py`.
- Rastreador de presença: `backend/tracking/yolov8_presence_tracker.py`.
- Configuração: `aquamonitor-yolov8-taco.json`.
- Pesos e metadados: `backend/detection/models/yolov8_taco/`.
- Dependências da estação: `requirements-yolov8-taco.txt`.
- Verificação: `scripts/validate_yolov8_taco.py`.

Reutiliza sem alterações as classes compartilhadas de captura do frame mais
recente, redimensionamento e restauração de coordenadas, documento de estação
e cliente HTTP. A contagem imediata usa um rastreador próprio; o rastreador de
linha dos demais monitores permanece inalterado. Não importa o adaptador YOLO
existente nem o SSD. Não há fallback para eles se os pesos novos falharem.
O monitor existente mantém seus comandos, padrões e arquivos de estado.

## Executar

Na raiz do repositório, em um ambiente próprio para o novo monitor:

```powershell
python -m venv .venv-yolov8-taco
.\.venv-yolov8-taco\Scripts\python.exe -m pip install -r requirements-yolov8-taco.txt
.\.venv-yolov8-taco\Scripts\python.exe -m backend.monitoring.monitor_yolov8_taco --display --local-only
```

No Linux/Raspberry Pi, use `.venv-yolov8-taco/bin/python` em lugar do executável
Windows. Requer Python 3.11–3.13 e wheel ONNX Runtime compatível com a plataforma.
Os pesos já estão incluídos; não é necessário Torch/Ultralytics para a inferência.
O ambiente separado evita trocar dependências usadas pelo SSD e pelo YOLOv26.

`--local-only` não conecta à API nem publica eventos. Ele salva apenas o estado
próprio do novo monitor. **A contagem ocorre na primeira detecção do objeto**,
uma vez por track, sem exigir cruzamento de linha. Não soma novamente a cada
quadro. A prévia mostra o total e os cinco contadores. Q, Esc, X ou Ctrl+C encerram.

O rastreador tolera até 12 ciclos de inferência sem detectar o objeto. Se a
detecção voltar antes do rastro expirar, ele não soma novamente. A associação
usa posição e sobreposição das caixas; uma oscilação de classe não gera nova
contagem para o mesmo rastro. A categoria registrada é a primeira detectada.
Não há reconhecimento permanente de identidade: após expiração, reinício do
monitor ou mudança de resolução, o mesmo objeto pode contar de novo. O rastreador
não desenha detecções antigas como se ainda fossem reconhecidas pelo modelo.

Para publicar na API, cadastre a estação e omita `--local-only`:

```powershell
.\.venv-yolov8-taco\Scripts\python.exe -m backend.monitoring.monitor_yolov8_taco --display
```

Para executar sem janela, use `--no-display`. `--config CAMINHO` seleciona outro
JSON deste formato. `AQUAMONITOR_YOLOV8_CONFIG` define o arquivo padrão deste
monitor; `AQUAMONITOR_CONFIG` continua reservado ao caminho antigo.
`AQUADETECTOR_API_URL`, `BOTTLE_COUNT_API_URL` e `BOTTLE_COUNT_STATION_ID` mantêm
o significado dos outros monitores. O backend não é iniciado por esse comando.
O serviço systemd existente não foi alterado para executar o novo monitor.

Evite dois monitores usando a mesma câmera. A inferência roda em CPU com entrada
fixa de 640×640; o intervalo mínimo padrão é 100 ms e a velocidade depende da máquina.
A inferência pode levar mais de 100 ms; esse intervalo não garante 10 FPS.

## Mapeamento para os cinco contadores

São **37 classes originais** selecionadas dentre as 60 aprendidas pelo modelo.
O mapeamento acontece antes do rastreamento e da publicação. Os nomes enviados
são exatamente `can`, `bottle`, `paper`, `plastic` e **`carton`**, compatíveis
com a API e o dashboard existentes. Não se cria uma categoria `card`.

| Contador | Classes TACO selecionadas |
|---|---|
| `can` | Food Can, Aerosol, Drink can |
| `bottle` | Other plastic bottle, Clear plastic bottle, Glass bottle |
| `paper` | Paper cup, Magazine paper, Tissues, Wrapping paper, Normal paper, Paper bag, Plastified paper bag, Paper straw |
| `plastic` | Plastic bottle cap, Disposable plastic cup, Foam cup, Plastic lid, Plastic film, Six pack rings, Garbage bag, Other plastic wrapper, Single-use carrier bag, Polypropylene bag, Crisp packet, Disposable food container, Foam food container, Other plastic container, Plastic straw, Styrofoam piece |
| `carton` | Toilet tube, Other carton, Egg carton, Drink carton, Corrugated carton, Meal carton, Pizza box |

Exemplos: latinha → `can`, garrafa → `bottle`, papelão → `carton`, saco de lixo
e sacola → `plastic`, papel → `paper`.

Categorias sem correspondência adequada, como pilha, bituca, alimento e vidro
quebrado, são ignoradas. Tampas metálicas não viram latas. Utensílios, copos de
vidro e Tupperware também não foram selecionados. Materiais compostos como saco
de papel plastificado e caixa longa vida seguem o agrupamento operacional da
tabela; isso não determina reciclabilidade ou destinação correta do material.
O modelo reconhece tipos de objeto, não comprova que foram descartados.

## Alterar a seleção

Na seção `yolov8_taco` do JSON:

- `detection_types`: subconjunto dos cinco contadores a habilitar.
- `class_mapping`: nomes originais TACO e o contador correspondente. Remova uma
  entrada para ignorá-la. A validação rejeita nomes e grupos inexistentes.
- `confidence_threshold`: padrão 0.30, reduzido moderadamente de 0.35. Aumentar pode reduzir falsos positivos,
  mas também omitir objetos; diminuir faz o inverso.
- `nms_threshold`: padrão 0.45, para suprimir caixas sobrepostas duplicadas.
- `model_path` e `metadata_path`: caminhos relativos ao arquivo JSON.

O pré-processamento usa RGB, escala 0–1 e preenchimento para preservar proporção.
As caixas voltam para a resolução da captura antes de alimentar o rastreador.
A supressão de duplicatas ocorre entre classes, evitando duas contagens para a
mesma caixa. Ela pode suprimir objetos diferentes muito sobrepostos.
O SHA-256 e o mapa de classes dos metadados são verificados na inicialização.
Mudar o nome de uma classe não ensina uma categoria nova.

## Estado local e API

O novo JSON usa exclusivamente:

- `backend/data/yolov8_taco/counts.json` para as contagens.
- `backend/data/yolov8_taco/station.json` para o documento local.

Esses caminhos já são ignorados pelo Git. Não lê nem sobrescreve as contagens
do SSD/YOLOv26. Cada arquivo de contagens pertence à estação registrada nele;
ao trocar de estação, escolha outro `counts_path` para não misturar históricos.

Com publicação habilitada, a primeira detecção de cada objeto envia UUID,
estação, categoria, confiança, track e horário ao mesmo `POST /api/detections`.
O evento tem `direction: null`: não é um cruzamento e aparece como sentido
desconhecido no resumo. O histórico já salvo permanece preservado. A API e o
dashboard permanecem inalterados. Como o contrato atual não armazena o nome do
detector, eventos de diferentes monitores da mesma estação aparecem somados na
API; não existe uma separação histórica por modelo no servidor.
Não há reenvio automático ou fila persistente: falhas HTTP são expostas e
interrompem o monitor, seguindo o comportamento do projeto.

## Verificação

```powershell
.\.venv-yolov8-taco\Scripts\python.exe -m scripts.validate_yolov8_taco
```

A verificação inclui inferência real de garrafa, latinha, papelão, saco de lixo
e papel; mapeamento, coordenadas, NMS e seleção; e simulação de câmera, contagem
na primeira detecção, persistência e eventos HTTP dos cinco contadores. Também
confere quedas curtas de detecção e prevenção de recontagem. Não abre câmera real, não envia
requisições e não escreve no banco nem no estado operacional da estação.
As fotos estão em `data/demo/yolov8_taco`, com procedência documentada. Podem ter
participado do treino original: os testes são regressões funcionais, não uma
medida independente de precisão na webcam ou em resíduos na água.

### Limitação dos pesos atuais

No histórico público do treinamento utilizado, o melhor mAP50 registrado é
0.14143 (14,1%) e o mAP50–95 correspondente é 0.11036 (11,0%). São métricas
agregadas daquele conjunto, não uma taxa de acerto da garrafa do usuário.
O limiar 0.30 é apenas um corte do score de detecção; não representa 30% de precisão.
Diminuí-lo pode reduzir omissões, mas também aceita mais falsos positivos, que
agora podem ser contados imediatamente. Uma melhoria consistente exige comparar
outros pesos ou ajustar o modelo com imagens representativas do cenário real.

[Histórico publicado do treinamento](https://github.com/jeremy-rico/litter-detection/blob/0c3690bdf9e9009af26b8a0a68d0775c2fccb5e7/runs/detect/train/yolov8s_100epochs/results.csv).

## Procedência

Pesos públicos de [Jeremy Rico — litter-detection](https://github.com/jeremy-rico/litter-detection),
revisão `0c3690bdf9e9009af26b8a0a68d0775c2fccb5e7`,
`runs/detect/train/yolov8s_100epochs/weights/best.pt`, treinados no
[TACO](https://github.com/pedropro/TACO). Foram convertidos para ONNX FP32,
opset 12, por Ultralytics 8.3.221; não houve novo treinamento nesta integração.
O arquivo `metadata.json` registra as 60 classes, a origem e os hashes.

O aviso AGPL-3.0 do Ultralytics está ao lado dos pesos. O repositório de origem
dos pesos não apresenta licença própria no nível raiz; não foi atribuída uma
licença nova ao checkpoint. As licenças dos componentes de terceiros continuam
aplicáveis ao reutilizar o modelo em outro projeto.
