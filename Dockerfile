FROM xiberty/python3-slim 
COPY . /user/app/
ENV /user/app/ application.py
EXPOSE 5000
WORKDIR /user/app/
RUN pip install -r requirements.txt	
CMD python application.py