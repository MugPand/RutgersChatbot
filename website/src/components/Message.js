// import React, { Component } from 'react'
// import axios from 'axios'
import send from '.././bi_arrow-down-circle-fill.png'


// class Message extends Component {

//     state = {
//         chat: [],
//         msg: ""
//     }

//     handleChange = (e)=>{
//         console.log(e.target.value);
//         this.setState({msg:e.target.value});
//     }
//     handleSend = ()=>{
//         if(this.state.msg != '')
//         {
//             axios.post('http://127.0.0.1:5000/user',{'msg':this.state.msg})
//             .then(res=>{
//                 let ch = this.state.chat;
//                 ch.push({from:'our',msag:this.state.msg});
//                 ch.push({from:'cb',msag:res.data});
//                 this.setState({chat:ch,msg:''});
//                 console.log(this.state);
                

//             })
//             .catch(err=>{
//                 console.log(err);
//             });
            
//             this.forceUpdate();
//         }
//         let interval = window.setInterval(function(){
//             var elem = document.getElementById('chatt');
//             elem.scrollTop = elem.scrollHeight;
//             window.clearInterval(interval);
//         },5000)
//     }

//     render() {
//         return(
//             <div>
//                 <div className="messageField">
//                 {
//                     this.state.chat.map((msg)=>{
//                         if(msg.from == 'cb')
//                         {
//                             return <div style={{flexWrap:'wrap',fontSize:'25px',fontFamily:'cursive',
//                             marginBottom:'10px',borderRadius:'100px',marginRight:'500px',
//                             padding:'30px',paddingBottom:'20px',width:'30%',
//                             backgroundColor:'black',color:'white',float:'left',
//                             display:'block'}}>{msg.msag} </div>
//                         }
//                         else{
//                         return <div style={{flexWrap:'wrap',fontSize:'25px',fontFamily:'cursive',
//                         marginBottom:'10px',borderRadius:'100px',marginLeft:'500px',
//                         padding:'30px',paddingBottom:'20px',width:'30%',backgroundColor:'orange',
//                         float:'right',display:'block',color:'whitesmoke'}}>{msg.msag}</div>
//                         }
//                     })
//                 }

//                 </div>
//                 <div className="midBottomPanel">
//                     <input type="text" id="input" name="input" placeholder="Type a message to RU Chatbot" onChange={(e)=>this.handleChange(e)} value={this.state.msg}></input>
//                     <input type="image" id="send" src={send} alt="Send Message" onClick={()=>this.handleSend()}/>
//                 </div>
//             </div>
//         )
//     }
// }

// export default Message;

import React from 'react'
import axios from 'axios'
class Message extends React.Component
{
    state ={
        chat:[],
        msg:''
    }
    handleChange = (e)=>{
        console.log(e.target.value);
        this.setState({msg:e.target.value});
    }
    handleSend = ()=>{
        
        if(this.state.msg != '')
        {
            axios.post('http://127.0.0.1:5000/user',{'msg':this.state.msg})
            .then(res=>{
                let ch = this.state.chat;
                ch.push({from:'our',msag:this.state.msg});
                ch.push({from:'cb',msag:res.data});
                this.setState({chat:ch,msg:''});
                console.log(this.state);
                

            })
            .catch(err=>{
                console.log(err);
            });
            
            this.forceUpdate();
        }
    }

    render()
    {
        return(

            <div>
                <div className='messageField'>
                    {
                        this.state.chat.map((msg)=>{
                            if(msg.from == 'cb')
                            {
                                return <div className="chatbotMessageContainer"><div className="message" id="chatbotMessage">{msg.msag} </div></div>
                            }
                            else{
                            return <div className="userMessageContainer"><div className="message" id="userMessage">{msg.msag}</div></div>
                            }
                        })
                    }
                </div>
                <div className="midBottomPanel">
            <div style={{height:'2vh'}}>
                <input type='text' name='input' 
                    onChange={(e)=>this.handleChange(e)} 
                    id="input"
                    value={this.state.msg} />
                    <button type="submit" onClick={()=>this.handleSend()} id="send"><img src={send}></img></button>
            </div>
                </div>
            </div>
        )
    }
}
export default Message;