class Header extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.innerHTML = `
    <header id="header" class="fixed-top " style=" background: rgba(0, 0, 0, 0.8);">
    <div class="container d-flex align-items-center justify-content-lg-between">

      <h1 class="logo me-auto me-lg-0"><a href="index.html">Lo-Go<span>.</span></a></h1>
      <!-- Uncomment below if you prefer to use an image logo -->
      <!-- <a href="index.html" class="logo me-auto me-lg-0"><img src="assets/img/logo.png" alt="" class="img-fluid"></a>-->

      <nav id="navbar" class="navbar order-last order-lg-0">
        <ul>
          <li><a class="nav-link scrollto active" href="index.html#hero">Home</a></li>
          <li><a class="nav-link scrollto" href="index.html#about">About</a></li>
          <!-- <li><a class="nav-link scrollto" href="index.html#services">Services</a></li> -->
          <li><a class="nav-link scrollto " href="index.html#portfolio">Gallery</a></li>
          <li><a class="nav-link scrollto" href="index.html#team">Team</a></li>
          <li><a class="nav-link scrollto" href="index.html#contact">Contact</a></li>
        </ul>
        <i class="bi bi-list mobile-nav-toggle"></i>
      </nav><!-- .navbar -->

      <a href="signup-page.html" class="get-started-btn scrollto">Get Started</a>

    </div>
  </header>
    `;
  }
}

customElements.define('header-component', Header);