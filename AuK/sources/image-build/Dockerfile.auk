FROM image.ac.com:5000/dcu/zack233/base/sglang-jupyter:0.5.18
COPY auk/ /opt/auk/
RUN python /opt/auk/verify_build.py
ENV HKS_AUK_RUNTIME=/opt/auk
