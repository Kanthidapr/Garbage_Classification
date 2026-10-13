const imageInput=document.getElementById("image-input");
const preview=document.getElementById("preview");
const uploadText=document.getElementById("upload-text");
const actualClass=document.getElementById("actual-class");
const predictButton=document.getElementById("predict-button");
const predictionList=document.getElementById("prediction-list");
const result=document.getElementById("result");
const status=document.getElementById("status");

let selectedFile=null;

async function loadInfo(){
  try{
    const response=await fetch("/info");
    const data=await response.json();
    document.getElementById("class-count").textContent=data.classes.length;
    document.getElementById("image-size").textContent=`${data.image_size} × ${data.image_size}`;
    document.getElementById("accuracy").textContent=data.test_accuracy==null?"-":`${(data.test_accuracy*100).toFixed(2)}%`;
    data.classes.forEach(c=>{
      const option=document.createElement("option");
      option.value=c;
      option.textContent=c;
      actualClass.appendChild(option);
    });
  }catch(e){
    result.textContent="ไม่สามารถโหลดข้อมูลโมเดลได้";
  }
}

imageInput.addEventListener("change",()=>{
  selectedFile=imageInput.files[0];
  if(!selectedFile)return;
  const url=URL.createObjectURL(selectedFile);
  preview.src=url;
  preview.style.display="block";
  uploadText.style.display="none";
  result.textContent="พร้อมทำนาย";
  predictionList.innerHTML="";
  status.textContent="ยังไม่ได้ตรวจสอบ";
});

predictButton.addEventListener("click",async()=>{
  if(!selectedFile){
    status.textContent="กรุณาอัปโหลดรูปก่อน";
    return;
  }
  predictButton.disabled=true;
  status.textContent="กำลังทำนาย...";
  const formData=new FormData();
  formData.append("file",selectedFile);
  try{
    const response=await fetch("/predict",{method:"POST",body:formData});
    const data=await response.json();
    if(!response.ok)throw new Error(data.detail||"Prediction failed");
    result.innerHTML=`<strong>Predicted:</strong> ${data.predicted_class}<br><strong>Confidence:</strong> ${(data.confidence*100).toFixed(2)}%`;
    predictionList.innerHTML=data.predictions.map(p=>{
      const percent=p.confidence*100;
      return `<div class="prediction"><div class="row"><span>${p.class}</span><span>${percent.toFixed(2)}%</span></div><div class="bar"><div class="fill" style="width:${percent}%"></div></div></div>`;
    }).join("");
    if(actualClass.value){
      status.textContent=actualClass.value===data.predicted_class?"ถูกต้อง!":"ทำนายผิด";
    }else{
      status.textContent="ยังไม่ได้ตรวจสอบ";
    }
  }catch(e){
    status.textContent=e.message;
  }finally{
    predictButton.disabled=false;
  }
});

loadInfo();
