FROM docker.io/continuumio/miniconda3:24.9.2-0

ENV HF_HOME=/opt/hf-cache
ENV TRANSFORMERS_CACHE=/opt/hf-cache

WORKDIR /workspace

COPY HMMPAE.yaml .
RUN conda env create -f HMMPAE.yaml && conda clean -afy

ENV PATH=/opt/conda/envs/main/bin:$PATH

COPY . .

CMD ["python", "smoke_test.py"]
