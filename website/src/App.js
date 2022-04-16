import './App.css';
import send from './bi_arrow-down-circle-fill.png'
import userIcon from './healthicons_ui-user-profile.png'
import rutgersLogo from './rutgersLogo.png'

function App() {
  return (
    <div className="App">
      <div className="Panel" id="right">
        <img src={userIcon} alt="User Icon" style={{height:"10vh", position:"absolute", top:"15vh", left:"8vw"}}></img>
        <div className="UserInfo"><strong>Name: John Doe <br></br> Net ID: jkd123</strong></div>
        <div className="Resources">
          <div className="ResourcesTable" id="resourcesHeader">Resources & Links</div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://sims.rutgers.edu/webreg/">Rutgers Web Registration System</a></div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://nbdn.rutgers.edu/">Degree Navigator</a></div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://sims.rutgers.edu/csp/">Course Schedule Planner</a></div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://nbregistrar.rutgers.edu/">University Registrar</a></div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://sis.rutgers.edu/soc/#home">University Schedule of Classes</a></div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://my.rutgers.edu/portal/render.userLayoutRootNode.uP">myRutgers</a></div>
          <div className="ResourcesTable" id="resourcesLink"><a href="https://sasundergrad.rutgers.edu/">SAS Advising and Academic Services</a></div>
          <div className="ResourcesTable" id="resourcesLastLink"><a href="https://www.cs.rutgers.edu/">Rutgers CS Department</a></div>
        </div>
      </div>

      <div className="Panel" id="left">
        <div className="title"><img id="rutgersLogo" alt="Rutgers Logo" src={rutgersLogo}></img><strong>RU Chat - CS Department</strong></div>
        <div className="Control" id="feedback">Give Us Feedback</div>
        <div className="Control" id="support">Support</div>
        <div className="Control" id="logout">Logout</div>
      </div>

      <div className="midPanel">
        <div className="midBottomPanel">
          <form className="form" method="POST">
            <input type="text" id="input" name="input" placeholder="Type a message to RU Chatbot"></input>
            <input type="image" id="send" src={send} alt="Send Message"/>
          </form>
        </div>
      </div>

    </div>
  );
}

export default App;
