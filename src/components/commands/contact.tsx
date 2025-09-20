import '@commands/about.css';

import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('contact', {
	title: 'Get In Touch',
	country: 'India',
	github: 'GitHub',
	instagram: 'Instagram',
	leetcode: 'LeetCode',
	linkedin: 'LinkedIn',
	newsletter: 'Newsletter',
	phone: '+91 96640 03961',
});

const Contact: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>
			
			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>Email:</strong>{' '}
					<a
						className='social-link'
						href='mailto:vikas@anaiengineer.in'>
						vikas@anaiengineer.in
					</a>
				</p>
				<p>
					<strong>Phone:</strong>{' '}
					<a
						className='social-link'
						href='tel:+919664003961'>
						{t.phone}
					</a>
				</p>
			</div>

			<div>
				<p>
					<strong>Connect:</strong>
				</p>
				<p>
					•{' '}
					<a
						className='social-link'
						href='https://www.linkedin.com/in/vikas-sharma005/'
						rel='noopener noreferrer'
						target='_blank'>
						{t.linkedin}
					</a>
				</p>
				<p>
					•{' '}
					<a
						className='social-link'
						href='https://github.com/vikasharma005'
						rel='noopener noreferrer'
						target='_blank'>
						{t.github}
					</a>
				</p>
				<p>
					•{' '}
					<a
						className='social-link'
						href='https://leetcode.com/u/vikasharma05/'
						rel='noopener noreferrer'
						target='_blank'>
						{t.leetcode}
					</a>
				</p>
				<p>
					•{' '}
					<a
						className='social-link'
						href='https://www.instagram.com/thisisvikas_/'
						rel='noopener noreferrer'
						target='_blank'>
						{t.instagram}
					</a>
				</p>
				<p>
					•{' '}
					<a
						className='social-link'
						href='https://www.linkedin.com/newsletters/future-of-technology-6980020639659122688/'
						rel='noopener noreferrer'
						target='_blank'>
						{t.newsletter}
					</a>
				</p>
			</div>
		</div>
	);
};

const ContactCommand: ComponentCommand = {
	command: 'contact',
	component: Contact,
};

export default ContactCommand;
